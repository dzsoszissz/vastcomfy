#!/bin/bash
###############################################################################
# Vast.ai provisioning szkript — chboishabba/WhisperX-WebUI
#
# Alapelvek (lásd VastAI_WhisperX-WebUI_Specifikacio_v4.md):
#   - a gyári CUDA/PyTorch környezet NEM kerül újratelepítésre
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
#   - a szolgáltatást a Start Command indítja, ez a szkript nem indít semmit
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
# ahonnan a szolgáltatás induláskor (Start Command) semmit nem érne el.
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
# 2b. Kompatibilitási folt: torch.load weights_only (PyTorch 2.6+)
#
# PyTorch 2.6-tól a torch.load() alapértelmezetten weights_only=True módban
# fut (biztonsági célból). A pyannote szegmentációs modellje egy régebbi
# PyTorch-Lightning-mentésű checkpoint, ami ezzel a szigorú móddal nem
# tölthető be (UnpicklingError egy torch.torch_version.TorchVersion
# globálra). Mivel hivatalos, HF_TOKEN-nel hitelesített, trusted forrásból
# (huggingface.co/pyannote) töltjük le, biztonságosan visszaállíthatjuk
# weights_only=False-ra. Ez torch-verzió miatti, nem pyannote-verzió miatti
# probléma, tehát verzió-pinneléssel nem küszöbölhető ki — marad a
# forráskód-szintű folt, ugyanabban a fájlban és mintában, ahol a repo
# saját maga is hasonlót alkalmaz a torchaudio.AudioMetaData hiányára.
# Mivel ez a fájl a "git reset --hard"-tól minden futásnál pristine
# állapotból indul, ezt a foltot minden provisioning-futásnál újra
# alkalmazni kell — ezért itt, a checkout után azonnal.
###############################################################################
echo "--- 2b. Kompatibilitási folt (torch.load weights_only) ---"
python - <<PYEOF
path = "${REPO_DIR}/modules/diarize/diarize_pipeline.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

marker = "from pyannote.audio import Pipeline"
patch = '''import torch as _torch

if not getattr(_torch.load, "_weights_only_compat", False):
    _original_torch_load = _torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_torch_load(*args, **kwargs)

    _torch_load_compat._weights_only_compat = True
    _torch.load = _torch_load_compat

'''

if marker not in content:
    raise SystemExit(f"Nem talalhato a vart sor a diarize_pipeline.py-ban: {marker!r}")

if "_weights_only_compat" not in content:
    content = content.replace(marker, patch + marker, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("  diarize_pipeline.py befoltozva (torch.load weights_only kompatibilitas)")
else:
    print("  diarize_pipeline.py mar tartalmazza a foltot")
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

# A requirements.txt sem a pyannote.audio-t, sem a huggingface_hub-ot nem
# pinneli verzióhoz. A pyannote.audio 4.0 elhagyta a "use_auth_token"
# paramétert (ez az upstream m-bain/whisperX projekt saját, hivatalosan
# dokumentált, ismert törése is — issue #1241: "pyannote 4 has breaking
# changes including use_auth_token", ajánlott workaround: pyannote-audio
# pinnelése 4.0 alá). Ahelyett hogy ezt monkeypatch-csel kerülnénk meg,
# egyszerűen egy még kompatibilis verzióra rögzítjük.
echo "pyannote.audio és huggingface_hub rögzítése kompatibilis verzióra..."
python -m pip install "pyannote.audio==3.4.0" "huggingface_hub<1.0"

echo "Gyári torch/CUDA build ellenőrzése (nem szabadott módosulnia):"
python - <<'PYEOF'
import torch
print(f"  torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
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

echo "  pyannote/speaker-diarization-3.1 (diarizáció, segmentation-3.0-t is behúzza)..."
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
python - <<PYEOF
import sys
sys.path.insert(0, "${REPO_DIR}")
from modules.uvr.music_separator import MusicSeparator

sep = MusicSeparator(model_dir="${UVR_MODEL_DIR}", output_dir="${OUTPUT_DIR}")
sep.update_model(model_name="UVR-MDX-NET-Inst_HQ_4", device="cpu")
PYEOF

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
yaml.preserve_quotes = True

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

echo "=== Provisioning kész. A szolgáltatást a Vast.ai Start Command indítja. ==="
