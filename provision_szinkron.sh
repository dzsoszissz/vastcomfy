#!/bin/bash
###############################################################################
# Vast.ai provisioning szkript — chboishabba/WhisperX-WebUI
#
# Alapelvek (lásd VastAI_WhisperX-WebUI_Specifikacio_v4.md):
#   - EZ A SZKRIPT A vastai/pytorch:2.8.0-cuda-12.8.1-py310-24.04-2026-06-15
#     IMAGE-HEZ KÉSZÜLT (nem a korábbi 2.7.1-hez, es a korabbi 12.6.3-tag
#     helyett most 12.8.1). A friss torch 2.8.0
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
# 2d. modules/vad/silero_vad.py javitasa — VALOS HIBA ALAPJAN (2026-08).
#
# A repo sajat get_speech_timestamps()-je a hangot SZANDEKOSAN 2D tombbe
# alakitja hivas elott ("padded_audio = np.expand_dims(padded_audio,
# axis=0)"), egy komment szerint mert "faster-whisper VAD expects a 2D
# array shaped as (batch_size, num_samples)". EZ HIBAS FELTETELEZES a
# nalunk telepitett faster-whisper verziohoz kepest — annak sajat VAD-
# modellje (faster_whisper/vad.py) kifejezetten 1D tombot var:
# "assert audio.ndim == 1, Input should be a 1D array". Ez pontosan
# ugyanaz a fajta verzio-elteresi hiba, mint a diarize_pipeline.py-nal —
# a repo befagyott kodja mas konvenciot feltetelez, mint a ténylegesen
# telepitett konyvtar aktualis viselkedese. A javitas egyszeruen
# eltavolitja a felesleges/hibas expand_dims hivast, igy a hang 1D marad,
# ahogy a tenyleges faster-whisper VAD elvarja.
#
# Mivel ez a fajl is a "git reset --hard"-tol minden futasnal pristine
# allapotbol indul, ezt minden provisioning-futasnal ujra kell alkalmazni.
###############################################################################
echo "--- 2d. silero_vad.py javitasa (felesleges 2D-atalakitas eltavolitasa) ---"
python - <<PYEOF
path = "${REPO_DIR}/modules/vad/silero_vad.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_expand = '''        padded_audio = np.pad(
            audio, (0, window_size_samples - audio.shape[0] % window_size_samples)
        )
        # faster-whisper VAD expects a 2D array shaped as (batch_size, num_samples)
        padded_audio = np.expand_dims(padded_audio, axis=0)
        # Guard against scalar outputs for very short inputs.'''
new_expand = '''        padded_audio = np.pad(
            audio, (0, window_size_samples - audio.shape[0] % window_size_samples)
        )
        # JAVITVA (provision.sh patch, 2026-08): a telepitett faster-whisper
        # VAD-modellje 1D tombot var (assert audio.ndim == 1), NEM 2D-t — a
        # kovetkezo sor eredetileg feltetelezte hogy 2D (batch, samples)
        # kell, de ez verzio-elteres miatt "AssertionError: Input should
        # be a 1D array"-t okozott. Eltavolitva, padded_audio 1D marad.
        # Guard against scalar outputs for very short inputs.'''
if old_expand not in content:
    raise SystemExit(f"Nem talalhato a vart sor a silero_vad.py-ban: {old_expand!r}")
content = content.replace(old_expand, new_expand, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("  silero_vad.py javitva (felesleges np.expand_dims eltavolitva, 1D marad a hang)")
PYEOF

###############################################################################
# 2c. Saját router hozzáadása a MEGLÉVŐ backendhez: /translate, /tts, /health
#
# Nem külön szolgáltatás/port — a WhisperX-WebUI-backend SAJÁT
# backend/main.py-jába kötjük be, ugyanúgy, ahogy a repo saját routerei
# (transcription_router, bgm_separation_router stb.) be vannak kötve. Ez
# elkerüli a külön port Vast.ai-portlistába vételének problémáját.
#
# A router.py fájlt a repo saját backend/routers/ struktúrájába írjuk
# (mint egy új alkönyvtár), a main.py-t pedig egy import + include_router
# sorral bővítjük. Mivel mindkettő a repo része, a "git reset --hard"
# minden futásnál töröl(het)i/visszaállítja őket — ezért ennek a lépésnek
# is a checkout UTÁN, a backend indítása (7. lépés) ELŐTT kell futnia,
# minden provisioning-futásnál újra.
#
# FONTOS VRAM-megfontolás: a Qwen3-Stylizer/F5-TTS modelleket
# SZÁNDÉKOSAN NEM tesszük be a main.py lifespan()-jének eager-load
# listájába (ahogy a whisper/vad/bgm-separation modellek be vannak) —
# azok @functools.lru_cache miatt csak az ELSŐ /translate vagy /tts
# híváskor töltődnek be. Ha eager lenne, a szerver induláskor egyszerre
# próbálná VRAM-ba tölteni az egész whisper+diarizáció+UVR+forditas+TTS
# stacket, ami könnyen túlcsordulhatna a 24 GB-os 3090-en.
###############################################################################
echo "--- 2c. Saját /translate, /tts router beillesztése a meglévő backendbe ---"

mkdir -p "${REPO_DIR}/backend/routers/custom_ai"
touch "${REPO_DIR}/backend/routers/custom_ai/__init__.py"

# A router.py mostantol KULON, felhasznalo altal hosztolt GitHub-repobol
# toltodik le (user kerese, 2026-08) — nem beagyazott heredoc a
# provision.sh-ban, igy konnyebb kulon szerkeszteni/verziozni, es a
# tenyleges forrast nem kell duplikalni ket helyen.
ROUTER_PY_URL="https://raw.githubusercontent.com/dzsoszissz/vastcomfy/refs/heads/main/szinkron_router.py"
curl -fsSL "$ROUTER_PY_URL" -o "${REPO_DIR}/backend/routers/custom_ai/router.py"
if [ $? -ne 0 ] || [ ! -s "${REPO_DIR}/backend/routers/custom_ai/router.py" ]; then
    echo "HIBA: nem sikerult letolteni a router.py-t innen: $ROUTER_PY_URL" >&2
    exit 1
fi
echo "  router.py letoltve: $ROUTER_PY_URL"

python - <<PYEOF
path = "${REPO_DIR}/backend/main.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_import = "from backend.routers.task.router import task_router"
new_import = (
    "from backend.routers.task.router import task_router\n"
    "from backend.routers.custom_ai.router import custom_ai_router"
)
if old_import not in content:
    raise SystemExit(f"Nem talalhato a vart import-sor a main.py-ban: {old_import!r}")
content = content.replace(old_import, new_import, 1)

old_include = "app.include_router(task_router)"
new_include = "app.include_router(task_router)\napp.include_router(custom_ai_router)"
if old_include not in content:
    raise SystemExit(f"Nem talalhato a vart include_router sor a main.py-ban: {old_include!r}")
content = content.replace(old_include, new_include, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("  main.py javitva (custom_ai_router bekotve)")
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
# FONTOS (valos hiba alapjan hozzaadva): a "reread"+"update" ONLY az UJ
# vagy megvaltozott .conf-fajlokat veszi eszre — ha a whisperx-backend MAR
# fut egy korabbi provisioning-futasbol, a folyamat memoriajaban meg a
# REGI router.py van betoltve, meg akkor is, ha a lemezen mar a friss
# valtozat van. Explicit restart kell, KULONBEN a router.py-ban tett
# valtoztatasok (uj vegpontok, promptmodositasok stb.) nem lepnek
# ervenybe ujrafuttataskor — ez okozta a "/custom-ai/separate 404" hibat.
supervisorctl restart whisperx-backend
echo "  whisperx-backend regisztrálva és (újra)indítva a supervisor alatt (port 8000)"
echo "  napló: /var/log/portal/whisperx-backend.log"

###############################################################################
# 8. Fordítási modell: Qwen3.6-27B GGUF, llama.cpp/llama-server-en
#    keresztül (NEM a teljes bf16 + transformers/AutoProcessor út)
#
# TÖRTÉNET (2026-08): eredetileg NLLB-200+Racka-4B, majd NLLB+Qwen3-4B-
# Instruct-2507, majd Qwen3-14B, majd Qwen/Qwen3.6-27B teljes bf16 +
# transformers (~56 GB letöltés). A felhasználó kifejezetten elutasította
# az 56 GB-os letöltést ("Isten ments"), és GGUF-ra kérte a váltást — ez
# egyben ARCHITEKTÚRA-VÁLTÁS is: a modell mostantól NEM a whisperx-
# backend Python-folyamatában fut in-process, hanem egy KÜLÖN
# llama-server folyamatként, OpenAI-kompatibilis HTTP API-val (saját
# supervisor-program, port 8002). A router.py translate_batch_with_
# context()-je csak HTTP-hívást csinál ehhez.
#
# MIÉRT GGUF ÉS NEM AWQ: kerestünk kb. 16 GB-os, előre kvantált Qwen3.6-
# 27B verziókat is (AWQ/INT4) — de ezek MIND harmadik felesek, és a Qwen
# saját HF-fórumán is panaszkodtak rájuk ("not stable to use"). A GGUF
# (unsloth/Qwen3.6-27B-GGUF, Q4_K_M, ~16.8 GB) sokkal érettebb,
# köztiszteletben álló formátum — az unsloth és bartowski kiadásait
# külön KL-divergencia-benchmarkkal is összemérték más kiadókkal.
#
# ISMERT HIBA (dokumentált): unsloth saját dokumentációja szerint CUDA
# 13.2-vel a Qwen3.6 GGUF értelmetlen ("gibberish") kimenetet adhat.
# ELLENORIZVE (2026-08): a kep idokozben CUDA 12.8.1-re valtozott (nem
# 12.6.3, ahogy korabban itt allt) — ez MEG MINDIG messze a problemas
# 13.2 alatt van, tehat ez a konkret hiba nem erint. HA a kep legkozelebb
# is valtozik, EZT UJRA ELLENORIZNI KELL.
#
# A modellt magát a llama-server "-hf" kapcsolója tölti le AUTOMATIKUSAN,
# első induláskor, közvetlenül a HuggingFace-ről — nincs külön
# snapshot_download lépés.
#
# FONTOS: ez a lepes SZANDEKOSAN nem szakitja meg a szkriptet hiba eseten
# (set +e / set -e keretezve), es a llama-server ONMAGABAN, KULON
# supervisor-programkent fut — ha az epites/inditas elhasal, a
# whisperx-backend (7. lepes) attol meg mukodik tovabb.
###############################################################################
echo "--- 8. llama.cpp építése + Qwen3.6-27B GGUF (Q4_K_M, ~16.8 GB) ---"

set +e

apt-get install -y --no-install-recommends cmake build-essential libcurl4-openssl-dev
if [ $? -ne 0 ]; then
    echo "  FIGYELEM: a llama.cpp build-fuggosegek telepitese nem sikerult." >&2
fi

LLAMA_CPP_DIR="${WORKSPACE_DIR}/llama.cpp"
LLAMA_PREBUILT_DIR="${WORKSPACE_DIR}/llama.cpp-prebuilt"
LLAMA_PREBUILT_OK=0

# ELOSZOR: kesz CUDA-bininaris letoltese (ai-dock/llama.cpp-cuda) — a
# hivatalos ggml-org repo NEM ad ki CUDA-binarist Linuxra, ez a projekt
# ezt a hianyt poltolja. A kepunk CUDA 12.8.1, ehhez a "cuda-12.8-amd64"
# csomag illik. Csak akkor esunk vissza forrasbol epitesre, ha a letoltes
# vagy a tenyleges futtatas sikertelen.
echo "  Kesz CUDA-bininaris probaja (ai-dock/llama.cpp-cuda, CUDA 12.8)..."
mkdir -p "$LLAMA_PREBUILT_DIR"
LLAMA_ASSET_URL=$(curl -s https://api.github.com/repos/ai-dock/llama.cpp-cuda/releases/latest \
    | grep "browser_download_url" | grep "cuda-12.8-amd64" | cut -d '"' -f 4 | head -1)

if [ -n "$LLAMA_ASSET_URL" ]; then
    echo "  Talalt csomag: $LLAMA_ASSET_URL"
    curl -fsSL "$LLAMA_ASSET_URL" -o "${LLAMA_PREBUILT_DIR}/llama-cuda.tar.gz"
    if [ $? -eq 0 ]; then
        tar -xzf "${LLAMA_PREBUILT_DIR}/llama-cuda.tar.gz" -C "$LLAMA_PREBUILT_DIR"
        LLAMA_SERVER_CANDIDATE=$(find "$LLAMA_PREBUILT_DIR" -type f -name "llama-server" | head -1)
        if [ -n "$LLAMA_SERVER_CANDIDATE" ]; then
            chmod +x "$LLAMA_SERVER_CANDIDATE"
            # Tenyleges futtatasi teszt — nem eleg, hogy letoltodott, tenyleg
            # el is kell induljon (pl. hianyzo CUDA-runtime-lib nem mindig
            # derul ki puszta letezesbol).
            if "$LLAMA_SERVER_CANDIDATE" --version > /tmp/llama_prebuilt_test.log 2>&1; then
                echo "  Kesz bininaris MUKODIK: $LLAMA_SERVER_CANDIDATE"
                LLAMA_SERVER_BIN="$LLAMA_SERVER_CANDIDATE"
                LLAMA_PREBUILT_OK=1
            else
                echo "  FIGYELEM: a kesz bininaris letoltodott, de nem futtathato (lasd /tmp/llama_prebuilt_test.log) — visszaeses forrasbol epitesre." >&2
                cat /tmp/llama_prebuilt_test.log >&2
            fi
        else
            echo "  FIGYELEM: a letoltott csomagban nem talalhato llama-server — visszaeses forrasbol epitesre." >&2
        fi
    else
        echo "  FIGYELEM: a kesz bininaris letoltese nem sikerult — visszaeses forrasbol epitesre." >&2
    fi
else
    echo "  FIGYELEM: nem talalhato megfelelo kesz csomag (cuda-12.8-amd64) — visszaeses forrasbol epitesre." >&2
fi

if [ "$LLAMA_PREBUILT_OK" -eq 0 ]; then
    echo "  Forrasbol epites (visszaeses)..."
    if [ -d "${LLAMA_CPP_DIR}/.git" ]; then
        echo "  llama.cpp mar letezik, frissites..."
        git -C "$LLAMA_CPP_DIR" pull
    else
        git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
    fi

    if [ -d "$LLAMA_CPP_DIR" ]; then
        cmake -B "${LLAMA_CPP_DIR}/build" -S "$LLAMA_CPP_DIR" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
        cmake --build "${LLAMA_CPP_DIR}/build" --config Release -j"$(nproc)" --target llama-server
        if [ $? -ne 0 ]; then
            echo "  FIGYELEM: a llama.cpp (llama-server) epitese nem sikerult — a forditasi funkcio nem lesz elerheto, de a tobbi szolgaltatas fut tovabb." >&2
        fi
        LLAMA_SERVER_BIN="${LLAMA_CPP_DIR}/build/bin/llama-server"
    else
        echo "  FIGYELEM: a llama.cpp klonozasa nem sikerult." >&2
        LLAMA_SERVER_BIN="${LLAMA_CPP_DIR}/build/bin/llama-server"
    fi
fi

LLAMA_GGUF_CACHE="${WORKSPACE_DIR}/models/llama-gguf-cache"
mkdir -p "$LLAMA_GGUF_CACHE"

# VALOS HIBA ALAPJAN HOZZAADVA (2026-08): user eszlelte, hogy a Qwen GGUF
# (~16.8 GB) letoltese a llama-server SAJAT -hf mechanizmusaval
# eszelosen lassu (45 perc alatt csak 5.4 GB). Kideritve: a fajl a
# HuggingFace UJ, Xet-alapu tarolasaval van feltoltve (nem sima Git-LFS),
# amihez a gyorsitott letoltes a "hf_xet" csomagot igenyli — ezt a
# llama.cpp SAJAT, C++ letoltoje valoszinuleg nem implementalja, es nyers,
# nem-gyorsitott HTTP-letoltesre esik vissza. MEGOLDAS: elore letoltjuk
# Python huggingface_hub-bal (ami tamogatja a Xet-gyorsitast) PONTOSAN
# abba a gyorsitotar-konyvtarba, amit a llama-server maga is hasznalna —
# igy amikor a llama-server elindul, mar KESZEN talalja, nem tolt le
# semmit ujra. Ha ez a lepes barmiert sikertelen, a llama-server sajat
# (lassabb, de mukodo) letoltese lesz a visszaeses.
echo "  Qwen GGUF elore letoltese (Xet-gyorsitassal, huggingface_hub)..."
pip install -q "huggingface_hub[hf_xet]" 2>&1 | tail -3
python3 - <<PYEOF
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")  # a hf_xet a preferalt, nem a regi hf_transfer
try:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="unsloth/Qwen3.6-27B-GGUF",
        filename="Qwen3.6-27B-Q4_K_M.gguf",
        cache_dir="${LLAMA_GGUF_CACHE}",
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"  Elore letoltve: {path}")
except Exception as exc:
    print(f"  FIGYELEM: elozetes letoltes sikertelen ({exc}) — a llama-server sajat letoltesere esunk vissza.")
PYEOF

if [ -x "$LLAMA_SERVER_BIN" ]; then
    # ROUTER MOD (2026-08, VALOS HIBA ALAPJAN, teljes atiras): korabban a
    # qwen-translate EGYETLEN modellkent, -hf kapcsoloval indult, ami
    # MOHON, AZONNAL a szerver inditasakor betolti a modellt VRAM-ba —
    # emiatt kellett a sajat, egyedi (nem dokumentalt) stop/start-tanc a
    # whisperx-backend korul, hogy a ket folyamat sose probaljon egyszerre
    # VRAM-ot foglalni. Ez a hivatalos llama-server ROUTER MOD-javal
    # kivalthato: a router folyamat MAGA konnyu, NEM tolt be semmit
    # inditaskor (models-autoload alapertelmezetten bekapcsolva = a
    # modell csak az ELSO TENYLEGES keresre toltodik be), a
    # --sleep-idle-seconds pedig utana is kiuriti tetlenul. Forras:
    # https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
    # ("load-on-startup: Controls whether the model loads automatically
    # when the server starts"; "--models-autoload ... default: enabled").
    #
    # Ezzel a whisperx-backend.sh es a provisioning kozotti teljes,
    # egyedi stop/start-vezenyles (nem dokumentalt, sajat talalmany volt)
    # MEGSZUNIK — a ket folyamat egyszerre, egymastol fuggetlenul
    # indulhat, mert a qwen-translate router-folyamata inditaskor nem
    # nyul a VRAM-hoz.
    QWEN_MODELS_INI="${WORKSPACE_DIR}/models/llama-configs/qwen-translate.ini"
    mkdir -p "$(dirname "$QWEN_MODELS_INI")"
    cat > "$QWEN_MODELS_INI" <<'INIEOF'
[qwen3.6-27b]
hf-repo = unsloth/Qwen3.6-27B-GGUF:Q4_K_M
n-gpu-layers = 999
ctx-size = 8192
flash-attn = on
sleep-idle-seconds = 6
reasoning = on
jinja = true
load-on-startup = false
INIEOF

    cat > /opt/supervisor-scripts/qwen-translate.sh <<EOF
#!/bin/bash
export LLAMA_CACHE="${LLAMA_GGUF_CACHE}"
# VALOS HIBA ALAPJAN HOZZAADVA (2026-08): a llama-server -hf/--hf-repo
# letoltese hivatalosan a HF_TOKEN kornyezeti valtozot hasznalja
# hitelesiteshez ("-hft, --hf-token TOKEN ... default: value from
# HF_TOKEN environment variable", forras: ggml-org/llama.cpp/tools/
# server/README.md). Ezt korabban SOSEM allitottuk be ebben a scriptben
# — a supervisord sajat kornyezete nem feltetlenul orokli a provisioning
# sajat, interaktiv shell-jenek HF_TOKEN-jet, ezert itt explicit
# exportaljuk (a provision.sh sajat futasa mar megkoveteli, hogy HF_TOKEN
# be legyen allitva, ld. szkript eleje). User eszlelte: a letoltes
# "eszelosen lassan" ment, mintha nem hasznalna a tokent.
export HF_TOKEN="${HF_TOKEN}"
exec ${LLAMA_SERVER_BIN} \\
    --host 0.0.0.0 --port 8002 \\
    --models-preset ${QWEN_MODELS_INI} \\
    --models-max 1
EOF
    chmod +x /opt/supervisor-scripts/qwen-translate.sh

    cat > /etc/supervisor/conf.d/qwen-translate.conf <<'EOF'
[program:qwen-translate]
command=/opt/supervisor-scripts/qwen-translate.sh
autostart=true
autorestart=true
startretries=3
stdout_logfile=/var/log/portal/qwen-translate.log
stdout_logfile_maxbytes=10MB
stderr_logfile=/var/log/portal/qwen-translate.log
stderr_logfile_maxbytes=10MB
EOF

    supervisorctl reread
    supervisorctl update
    supervisorctl restart qwen-translate
    echo "  qwen-translate (llama-server, ROUTER MOD) regisztrálva és elindítva a supervisor alatt (port 8002)"
    echo "  napló: /var/log/portal/qwen-translate.log"
    echo "  MEGJEGYZÉS: a router-folyamat maga konnyu, a GGUF (~16.8 GB) csak az ELSŐ tényleges fordítási kérésre töltődik be, majd 6 mp inaktivitás után automatikusan kiürül."
else
    echo "  FIGYELEM: nem talalhato a lefordított llama-server binaris ($LLAMA_SERVER_BIN) — a qwen-translate szolgaltatas nem lett regisztralva." >&2
fi

###############################################################################
# 9. F5-TTS (SWivid/F5-TTS, ComfyUI NELKUL, sima pip csomag + CLI/Python API)
#    + a Maxdorger29/f5-tts-hungarian magyar hangklonozo checkpoint.
#
# ##############################################################################
# ##  FIGYELEM — LICENC KORLATOZAS, NE FELEJTSD EL!                          ##
# ##                                                                          ##
# ##  A Maxdorger29/f5-tts-hungarian (es az alapja, SWivid/F5-TTS sajat      ##
# ##  sulyai) CC-BY-NC-4.0 licenc alatt vannak, MERT az Emilia adathalmazon  ##
# ##  lettek tanitva, ami kereskedelmi felhasznalast tilt.                   ##
# ##                                                                          ##
# ##  -> SZEMELYES / KUTATASI CEL: szabad hasznalni.                         ##
# ##  -> KERESKEDELMI CEL (ugyfelmunka, reklam, brand-videó stb.): TILOS      ##
# ##     ezekkel a sulyokkal, amig nincs kulon kereskedelmi licenc VAGY       ##
# ##     at nem valtunk mas licencu modellre (pl. XTTS-v2 / Coqui, ami       ##
# ##     tamogat magyart es mashogy licencelt).                              ##
# ##                                                                          ##
# ##  A jelenlegi celunk (2026-08) NEM kereskedelmi -> egyelore rendben.     ##
# ##  DE HA EZ VALTOZIK, ELOSZOR EZT A LEPEST KELL UJRAGONDOLNI!             ##
# ##############################################################################
#
# Maga az F5-TTS Python-csomag (nem a sulyok) MIT licencu, es ONALLOAN,
# ComfyUI nelkul is teljesen hasznalhato (f5-tts_infer-cli / Python API).
###############################################################################
echo "--- 9. F5-TTS + magyar checkpoint előtöltése (NEM KERESKEDELMI CÉLRA!) ---"

set +e

python -m pip install -q f5-tts
if [ $? -ne 0 ]; then
    echo "  FIGYELEM: az f5-tts csomag telepitese nem sikerult, a lepes kihagyva." >&2
fi

F5TTS_HU_DIR="${WORKSPACE_DIR}/models/F5-TTS-Hungarian"
mkdir -p "$F5TTS_HU_DIR"

python - <<PYEOF
from huggingface_hub import hf_hub_download

print("  Maxdorger29/f5-tts-hungarian letoltese (model + vocab)...")
hf_hub_download(
    repo_id="Maxdorger29/f5-tts-hungarian",
    filename="model_last_final.safetensors",
    local_dir="${F5TTS_HU_DIR}",
)
hf_hub_download(
    repo_id="Maxdorger29/f5-tts-hungarian",
    filename="vocab.txt",
    local_dir="${F5TTS_HU_DIR}",
)
PYEOF
if [ $? -ne 0 ]; then
    echo "  FIGYELEM: a magyar F5-TTS checkpoint elotoltese nem sikerult teljesen — a backend ettol meg fut. Probald ujra kezzel kesobb." >&2
fi

set -e

echo "  F5-TTS magyar checkpoint helye: ${F5TTS_HU_DIR}"
echo "  EMLEKEZTETO: ez a checkpoint CC-BY-NC-4.0 — csak nem-kereskedelmi hasznalatra!"
df -h / | tail -1

###############################################################################
# 9b. Magyar szoveg-normalizalo (sarpba/hun) — a /custom-ai/normalize_text
#     vegponthoz. NEM sajat iras — a tenyleges, valos repo klonozasa, hogy
#     mindig a hivatalos forras fusson, ne egy esetleg hibas masolat.
#     Forras: https://github.com/sarpba/hun
###############################################################################
echo "--- 9b. Magyar szoveg-normalizalo (sarpba/hun) telepitese ---"

set +e

HUN_NORMALIZER_DIR="${WORKSPACE_DIR}/models/hun-normalizer"
if [ -d "$HUN_NORMALIZER_DIR/.git" ]; then
    (cd "$HUN_NORMALIZER_DIR" && git pull)
else
    rm -rf "$HUN_NORMALIZER_DIR"
    git clone https://github.com/sarpba/hun.git "$HUN_NORMALIZER_DIR"
fi

python -m pip install -q num2words
if [ $? -ne 0 ]; then
    echo "  FIGYELEM: a num2words telepitese nem sikerult — a /custom-ai/normalize_text vegpont nem fog mukodni." >&2
fi

set -e

df -h / | tail -1

###############################################################################
# 10. audio-separator (nomadkaraoke) — BS-Roformer, a /custom-ai/separate
#     vegponthoz. Ez VALTJA FEL a WhisperX-WebUI beepitett /bgm-separation/
#     lepeset a run_pipeline.php-ban — annak sajat modell-listaja fixen
#     ['UVR-MDX-NET-Inst_HQ_4', 'UVR-MDX-NET-Inst_3']-ra korlatozodik
#     (ellenorizve elesben a MusicSeparator.available_models-szal), roformer
#     nincs benne, pedig 2026-ban a BS-Roformer a legjobb architektura ehhez
#     — a jelenlegi MDX-Net-es szetvalasztasnal szovegreszek vesztek el.
#
# A "model_bs_roformer_ep_317_sdr_12.9755.ckpt"-t hasznaljuk (nem a csomag
# sajat, Mel-Band-alapu default-jat) — ezt a csomag sajat karbantartoja
# ajanlja alapertelmezeskent tiszta, teljes spektrumu szetvalasztashoz a
# legtobb bemenethez (nomadkaraoke/python-audio-separator #133 discussion).
###############################################################################
echo "--- 10. audio-separator (BS-Roformer) telepítése és előtöltése ---"

set +e

python -m pip install -q "audio-separator[gpu]"
if [ $? -ne 0 ]; then
    echo "  FIGYELEM: az audio-separator[gpu] telepitese nem sikerult, a lepes kihagyva — a /custom-ai/separate vegpont nem lesz elerheto." >&2
fi

# Egyszeri elotoltes, hogy a modell mar most letoltodjon (ne az elso
# tenyleges hivas varjon ra) — a Separator sajat maga kezeli a HF-
# letoltest, nincs kulon snapshot_download hivas.
python - <<'PYEOF'
try:
    from audio_separator.separator import Separator
    print("  model_bs_roformer_ep_317_sdr_12.9755.ckpt eloltoltese...")
    sep = Separator()
    sep.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    print("  BS-Roformer sikeresen eloltoltve.")
except Exception as exc:
    print(f"  FIGYELEM: BS-Roformer elotoltese nem sikerult: {exc}")
PYEOF

set -e

df -h / | tail -1

echo "=== Provisioning kész. A backend a Vast.ai supervisor alatt fut (whisperx-backend, port 8000 — /transcription, /vad, /task, /custom-ai/separate, /custom-ai/translate_batch, /custom-ai/tts; qwen-translate, port 8002). ==="
