#!/bin/bash
###############################################################################
# Vast.ai provisioning szkript — chboishabba/WhisperX-WebUI
#
# Alapelvek (lásd VastAI_WhisperX-WebUI_Specifikacio_v4.md):
#   - EZ A SZKRIPT A vastai/pytorch:2.8.0-cuda-12.6.3-py310-24.04-2026-06-15
#     IMAGE-HEZ KÉSZÜLT (nem a korábbi 2.7.1-hez). A friss torch 2.8.0
#     gyárilag kielégíti a whisperx/pyannote.audio saját torch~=2.8.0
#     követelményét, így a pip nem próbál konfliktusos verziót erőltetni,
#     és a friss pyannote.audio (>=4.0) + huggingface_hub (>=0.28.1)
#     kombó natívan működik — nincs szükség monkeypatch-ekre.
#   - a repo saját Install.sh-ja NEM fut le (mert saját venv-et hozna
#     létre) — helyette a benne lévő 4 telepítési lépést futtatjuk
#     közvetlenül, a Vast.ai image gyári /venv/main-jében (lásd 0. lépés:
#     ez NEM azonos a PROVISIONING_SCRIPT saját, elkülönített
#     provisioner-venv-jével, amiben ez a szkript alapból elindul)
#   - a repo modell-/kimeneti útvonalai (modules/utils/paths.py) hardkódoltan,
#     a repo gyökeréhez képest relatívak — NEM environment változóból jönnek.
#     Ezért a repót magát klónozzuk a /workspace alá: így a hardkódolt
#     útvonalak automatikusan perzisztensek lesznek, nincs szükség
#     MODEL_DIR/INPUT_DIR/OUTPUT_DIR env-re vagy szimlinkre.
#   - a szolgáltatást ez a szkript regisztrálja és indítja el a Vast.ai
#     saját supervisor-jánál (7. lépés) — Jupyter/SSH launch módban a
#     docker image ENTRYPOINT-ja nem fut le, ezért nem elég egy sima
#     háttérfolyamat, a supervisorral kell integrálni az auto-restarthoz
###############################################################################

set -euo pipefail

echo "=== WhisperX-WebUI provisioning indul ==="

# --- Kötelező environment változó ellenőrzése ---
: "${HF_TOKEN:?HF_TOKEN nincs beállítva — a diarizációs modellhez kötelező}"

###############################################################################
# 0. A gyári /venv/main aktiválása
#
# FONTOS: a PROVISIONING_SCRIPT a Vast.ai saját, elkülönített
# "provisioner" venv-jében fut (pl. /opt/instance-tools/provisioner/venv),
# aminek SEMMI köze a konténer tényleges, torch/CUDA-t tartalmazó
# futtatókörnyezetéhez. Ha ezt nem aktiváljuk explicit, minden ezután
# következő "python"/"pip" hívás ebbe az üres, idegen venv-be telepítene,
# és a 7. lépésben regisztrált supervisor-szolgáltatás sem a helyes
# venv-ből indulna.
###############################################################################
echo "--- 0. Gyári venv aktiválása ---"
if [ -f /venv/main/bin/activate ]; then
    # shellcheck disable=SC1091
    source /venv/main/bin/activate
    echo "Aktiválva: /venv/main"
else
    echo "HIBA: /venv/main/bin/activate nem található — a provisioning leáll," >&2
    echo "mert a rendszer-python/pip a rossz (provisioner) venv-be telepítene." >&2
    exit 1
fi
echo "Aktív python: $(which python)"
echo "Aktív pip:    $(which pip)"

WORKSPACE_DIR="/workspace"
REPO_DIR="${WORKSPACE_DIR}/WhisperX-WebUI"
REPO_URL="https://github.com/chboishabba/WhisperX-WebUI"

FASTER_WHISPER_MODEL_DIR="${REPO_DIR}/models/Whisper/faster-whisper"
DIARIZATION_MODEL_DIR="${REPO_DIR}/models/Diarization"
UVR_MODEL_DIR="${REPO_DIR}/models/UVR/MDX_Net_Models"
OUTPUT_DIR="${REPO_DIR}/outputs"
BACKEND_CACHE_DIR="${REPO_DIR}/backend/cache"
BACKEND_CONFIG_PATH="${REPO_DIR}/backend/configs/config.yaml"

###############################################################################
# 1. Rendszercsomagok
###############################################################################
echo "--- 1. Rendszercsomagok telepítése ---"
apt-get update
apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    libportaudio2 \
    libsndfile1 \
    python3-filelock
rm -rf /var/lib/apt/lists/*

###############################################################################
# 2. Repository klónozása / frissítése (idempotens — új instance-nál a
#    /workspace megmarad, tehát ez git pull-ra egyszerűsödik)
###############################################################################
echo "--- 2. Repository ---"
mkdir -p "$WORKSPACE_DIR"

if [ -d "${REPO_DIR}/.git" ]; then
    echo "Repository már létezik, frissítés..."
    git -C "$REPO_DIR" fetch --all
    DEFAULT_BRANCH=$(git -C "$REPO_DIR" remote show origin | sed -n '/HEAD branch/s/.*: //p')
    git -C "$REPO_DIR" reset --hard "origin/${DEFAULT_BRANCH}"
else
    git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

###############################################################################
# 2b. Upstream-fix átvétele: use_auth_token -> token, ÉS modellváltás
#     speaker-diarization-3.1 -> speaker-diarization-community-1
#
# A repo modules/diarize/diarize_pipeline.py-ja egy 2023-24-ben befagyott
# másolata a https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py
# fájlnak. Az AKTUÁLIS upstream verzió már "token=token"-nel hívja a
# pyannote.audio Pipeline.from_pretrained()-jét — ez a repo befagyott
# másolata még a régi "use_auth_token="-t használja, ami a friss
# pyannote.audio (>=4.0) belső hf_hub_download()-hívásában TypeError-t dob.
#
# Nem monkeypatch-elünk (nincs rá szükség a friss torch 2.8.0 / pyannote.audio
# 4.x / huggingface_hub kombónál) — egyszerűen átvesszük az upstream saját,
# egysoros javítását a hívási helyen.
#
# MÁSODIK JAVÍTÁS: a "pyannote/speaker-diarization-3.1" a saját HF-oldala
# szerint egy "Legacy Collection, Compatible with pyannote.audio 3.4.x" —
# NEM a friss 4.x könyvtárhoz készült. A pyannote.audio 4.0.7 emiatt a 3.1
# betöltésekor is megpróbál egy PLDA-komponenst letölteni a friss
# "pyannote/speaker-diarization-community-1" repóból (belső könyvtár-logika,
# nem a 3.1 modell saját config.yaml-ja írja elő) — ez egy plusz, korábban
# nem elfogadott gated modellt igényelne. Mivel a community-1 kifejezetten
# a 4.x könyvtárhoz készült (és a HF-oldal szerint pontosabb is, mint a
# 3.1), a modell-alapértelmezést átírjuk rá — ezzel a PLDA már a "helyi",
# elvárt repóból jön, nem egy külön, nem dokumentált függőségként. A
# community-1 ráadásul a segmentation-t is saját magában tartalmazza, nem
# kell hozzá külön "pyannote/segmentation-3.0" elfogadás.
#
# Mivel ez a fájl a "git reset --hard"-tól minden futásnál pristine
# állapotból indul, ezt minden provisioning-futásnál újra kell alkalmazni.
###############################################################################
echo "--- 2b. use_auth_token -> token + community-1 modellváltás ---"
python - <<PYEOF
path = "${REPO_DIR}/modules/diarize/diarize_pipeline.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_token = "model_name, use_auth_token=use_auth_token, cache_dir=cache_dir"
new_token = "model_name, token=use_auth_token, cache_dir=cache_dir"
if old_token not in content:
    raise SystemExit(f"Nem talalhato a vart sor a diarize_pipeline.py-ban: {old_token!r}")
content = content.replace(old_token, new_token, 1)

old_model = 'model_name="pyannote/speaker-diarization-3.1"'
new_model = 'model_name="pyannote/speaker-diarization-community-1"'
if old_model not in content:
    raise SystemExit(f"Nem talalhato a vart sor a diarize_pipeline.py-ban: {old_model!r}")
content = content.replace(old_model, new_model, 1)

# Harmadik javitas: a pyannote.audio 4.x-ben a pipeline hivasa mar nem
# kozvetlenul egy Annotation-t (itertracks()-kepes objektumot) ad vissza,
# hanem egy DiarizeOutput dataclass-t, aminek a "speaker_diarization" mezoje
# tartalmazza a tenyleges Annotation-t (forras: pyannote/audio/pipelines/
# speaker_diarization.py, "class DiarizeOutput", "speaker_diarization:
# Annotation" mezo). A repo befagyott __call__-ja meg a regi, kozvetlen
# visszaterest feltetelezi -> AttributeError: 'DiarizeOutput' object has
# no attribute 'itertracks'. hasattr-rel vedekezunk, hatha egy regebbi
# pyannote-verzio meg kozvetlenul Annotation-t adna vissza.
old_call = '''        segments = self.model(
            audio_data, min_speakers=min_speakers, max_speakers=max_speakers
        )
        diarize_df = pd.DataFrame('''
new_call = '''        segments = self.model(
            audio_data, min_speakers=min_speakers, max_speakers=max_speakers
        )
        if hasattr(segments, "speaker_diarization"):
            segments = segments.speaker_diarization
        diarize_df = pd.DataFrame('''
if old_call not in content:
    raise SystemExit(f"Nem talalhato a vart sor a diarize_pipeline.py-ban: {old_call!r}")
content = content.replace(old_call, new_call, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("  diarize_pipeline.py javitva (use_auth_token -> token, model -> community-1, DiarizeOutput kicsomagolas)")
PYEOF

###############################################################################
# 3. Python függőségek — a repo saját telepítési lépései, rendszer-Pythonnal,
#    venv nélkül (nem az Install.sh fut, csak a benne lévő 4 sor)
###############################################################################
echo "--- 3. Python függőségek ---"
python -m pip install -U pip
python scripts/install_openai_whisper.py
python -m pip install -r requirements.txt
python -m pip install -r backend/requirements-backend.txt

# A requirements.txt nem pinneli a pyannote.audio-t verzióhoz. A TELJES
# függőségi gráf (a "uvr" csomag óriási tranzitív fájával, transformers-szel,
# a numpy<2.3 pinnel együtt) miatt a pip resolver ilyenkor visszalép egy
# régi, pyannote.audio==3.4.0 verzióra, aminek MÁS a hívási szignatúrája
# (use_auth_token=, nem token=), mint a friss 4.x-nek. Ez helyben, két
# teljes-gráfos "pip install --dry-run" futtatással ellenőrizve: a
# pyannote.audio==4.0.7 explicit kikényszerítése a TELJES requirements.txt
# mellett is konfliktusmentesen összeáll (numpy 2.2.6 marad <2.3, a torch
# a már telepített gyári verzióval elégül ki). A 4.0.7-et azért pont ezt
# választottuk, mert a forráskódjában nulla "use_auth_token" előfordulás
# van, és a modellbetöltés explicit weights_only=False-t ír elő magának.
echo "pyannote.audio rögzítése a teljes gráfon ellenőrzött 4.0.7 verzióra..."
python -m pip install "pyannote.audio==4.0.7"

echo "Torch/CUDA build ellenőrzése (2.8.0-t várunk, a friss image gyári torch-ja):"
python - <<'PYEOF'
import torch
print(f"  torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
PYEOF
echo "pyannote.audio verzió ellenőrzése (4.0.7-et várunk):"
python - <<'PYEOF'
import pyannote.audio
print(f"  pyannote.audio={pyannote.audio.__version__}")
PYEOF

###############################################################################
# 4. Cache / modell könyvtárak létrehozása (a /workspace alatt, tehát
#    perzisztens)
###############################################################################
echo "--- 4. Könyvtárstruktúra ---"
mkdir -p "$FASTER_WHISPER_MODEL_DIR"
mkdir -p "$DIARIZATION_MODEL_DIR"
mkdir -p "$UVR_MODEL_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$BACKEND_CACHE_DIR"

###############################################################################
# 5. Modellek előtöltése
#    Fontos: ugyanazokat a hívásokat/útvonalakat használjuk, amiket az app
#    saját maga is használna futásidőben (faster_whisper.WhisperModel,
#    pyannote Pipeline.from_pretrained, MusicSeparator.update_model) —
#    így később, első API-híváskor nem indul újabb letöltés.
###############################################################################
echo "--- 5. Modellek előtöltése ---"

echo "  faster-whisper large-v2 és large-v3..."
python - <<PYEOF
from faster_whisper import WhisperModel

model_dir = "${FASTER_WHISPER_MODEL_DIR}"
for size in ("large-v2", "large-v3"):
    print(f"    -> {size}")
    WhisperModel(size, download_root=model_dir, device="cpu", compute_type="int8")
PYEOF

echo "  pyannote/speaker-diarization-community-1 (diarizáció; a segmentation és embedding is ebben a repóban van, külön segmentation-3.0 elfogadás nem kell)..."
# Fontos: a repo SAJÁT modules/diarize/diarize_pipeline.py moduljának
# DiarizationPipeline osztályán keresztül töltjük elő, NEM közvetlen
# pyannote.audio importtal. A diarize_pipeline.py modul szinten
# monkey-patch-eli a torchaudio.AudioMetaData hiányát (torchaudio 2.8+
# esetén ez az attribútum eltűnt), és csak ez után importálja a
# pyannote.audio.Pipeline-t. Ha megkerüljük ezt a modult, a patch nem
# fut le, és AttributeError-ral elszáll a letöltés/betöltés.
python - <<PYEOF
import os
import sys

sys.path.insert(0, "${REPO_DIR}")
from modules.diarize.diarize_pipeline import DiarizationPipeline

DiarizationPipeline(
    cache_dir="${DIARIZATION_MODEL_DIR}",
    use_auth_token=os.environ["HF_TOKEN"],
    device="cpu",
)
PYEOF

echo "  UVR-MDX-NET-Inst_HQ_4 (BGM separation)..."
# MEGJEGYZÉS: a jhj0517/ultimatevocalremover_api csomagnak (amit a repo
# "uvr" néven használ) van egy dokumentált, ismert hibája (jhj0517/
# Whisper-WebUI issue #577): a modell-létezés ellenőrzésekor NEM a nekünk
# átadott model_dir-t nézi, hanem egy saját, a site-packages alá
# hardkódolt útvonalat — emiatt a modell újra letöltődhet minden
# instance-indításkor, FÜGGETLENÜL attól, hogy mi hova mentettük.
# Ez a modell kicsi (~60-120 MB, nem a 3 GB-os Whisper-modellek
# nagyságrendje), tehát ha ez be is következik, nem drága és nem
# blokkolja a szolgáltatást — de a lenti "find" kiírja a tényleges
# helyét, hogy lássuk, kell-e emiatt külön lépés.
python - <<PYEOF
import sys
sys.path.insert(0, "${REPO_DIR}")
from modules.uvr.music_separator import MusicSeparator

sep = MusicSeparator(model_dir="${UVR_MODEL_DIR}", output_dir="${OUTPUT_DIR}")
sep.update_model(model_name="UVR-MDX-NET-Inst_HQ_4", device="cpu")
PYEOF
echo "  UVR modellfájl(ok) tényleges helye a lemezen:"
find / -xdev -iname "*UVR-MDX-NET*" -not -path "*/proc/*" 2>/dev/null | sed 's/^/    /'

###############################################################################
# 6. Backend konfiguráció igazítása
#    A whisper modell/compute_type/device NEM environment változóból jön,
#    hanem a backend/configs/config.yaml-ból (lásd backend/common/config_loader.py
#    és backend/routers/transcription/router.py::get_pipeline) — ezért itt
#    közvetlenül a YAML-t módosítjuk. Kezdésnek large-v3, később a
#    model_size érték átírásával válthat a large-v2-re.
###############################################################################
echo "--- 6. backend/configs/config.yaml igazítása ---"
python - <<PYEOF
from ruamel.yaml import YAML

yaml = YAML(typ="safe")
yaml.map_indent = 2
yaml.sequence_indent = 4
yaml.sequence_dash_offset = 2
yaml.preserve_quotes = True
yaml.default_flow_style = False
yaml.sort_base_mapping_type_on_output = False

config_path = "${BACKEND_CONFIG_PATH}"
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.load(f)

cfg["whisper"]["model_size"] = "large-v3"
cfg["whisper"]["compute_type"] = "float16"
cfg["bgm_separation"]["device"] = "cuda"

with open(config_path, "w", encoding="utf-8") as f:
    yaml.dump(cfg, f)

print("  whisper.model_size = large-v3")
print("  whisper.compute_type = float16")
print("  bgm_separation.device = cuda")
PYEOF

###############################################################################
# 7. Backend szolgáltatás regisztrálása a Vast.ai supervisor-jánál
#
# FONTOS: Jupyter/SSH launch módban (amit ehhez a template-hez használunk)
# a Vast.ai a saját supervisor-jával (supervisord) menedzseli a konténerben
# futó szolgáltatásokat — a docker image saját ENTRYPOINT-ja NEM fut le
# ilyenkor (lásd Vast.ai "Advanced Setup" dokumentáció). Ha a backendet itt
# csak simán háttérbe indítanánk (pl. "uvicorn ... &"), a PROVISIONING_SCRIPT
# lefutása után az a folyamat elveszne / nem lenne auto-restart, ha
# összeomlana.
#
# A hivatalos minta szerint egy supervisor-program-ot kell regisztrálni:
# egy wrapper script /opt/supervisor-scripts/ alá, egy .conf fájl
# /etc/supervisor/conf.d/ alá, majd "supervisorctl reload".
###############################################################################
echo "--- 7. Backend regisztrálása a supervisor-nál ---"

mkdir -p /opt/supervisor-scripts

cat > /opt/supervisor-scripts/whisperx-backend.sh <<EOF
#!/bin/bash
source /venv/main/bin/activate
cd "${REPO_DIR}"
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000
EOF
chmod +x /opt/supervisor-scripts/whisperx-backend.sh

cat > /etc/supervisor/conf.d/whisperx-backend.conf <<'EOF'
[program:whisperx-backend]
command=/opt/supervisor-scripts/whisperx-backend.sh
autostart=true
autorestart=true
startretries=3
stdout_logfile=/var/log/portal/whisperx-backend.log
stdout_logfile_maxbytes=10MB
stderr_logfile=/var/log/portal/whisperx-backend.log
stderr_logfile_maxbytes=10MB
EOF

supervisorctl reread
supervisorctl update
echo "  whisperx-backend regisztrálva és elindítva a supervisor alatt (port 8000)"
echo "  napló: /var/log/portal/whisperx-backend.log"

echo "=== Provisioning kész. A backend a Vast.ai supervisor alatt fut (whisperx-backend, port 8000). ==="
