#!/bin/bash
set -euo pipefail

# Vast.ai on-start / provisioning script
# Target:
# - Existing Vast PyTorch CUDA 12.6.3 image
# - Python 3.10
# - Do NOT reinstall torch / torchaudio

APP_ROOT=/workspace/Whisper-WebUI
MODEL_ROOT=/workspace/models
HF_ROOT=/workspace/huggingface
OUTPUT_ROOT=/workspace/output

export HF_HOME=${HF_ROOT}
export TRANSFORMERS_CACHE=${HF_ROOT}
export CUDA_VISIBLE_DEVICES=0


apt update
apt install -y \
    git \
    ffmpeg


mkdir -p \
    /workspace \
    ${MODEL_ROOT}/Whisper \
    ${MODEL_ROOT}/Diarization \
    ${MODEL_ROOT}/UVR \
    ${OUTPUT_ROOT}


cd /workspace


if [ ! -d "${APP_ROOT}" ]; then
    git clone https://github.com/jhj0517/Whisper-WebUI.git
fi


cd ${APP_ROOT}


python3 -m pip install --upgrade pip setuptools wheel

pip install -r requirements.txt


if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN missing"
    exit 1
fi


huggingface-cli login --token "${HF_TOKEN}"


# faster-whisper large-v3 preload

python3 - <<'PY'
from faster_whisper import WhisperModel

WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16",
    download_root="/workspace/models/Whisper"
)

print("large-v3 OK")
PY


# pyannote preload

python3 - <<'PY'
import os
from pyannote.audio import Pipeline

token=os.environ["HF_TOKEN"]

Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=token
)

Pipeline.from_pretrained(
    "pyannote/segmentation-3.0",
    use_auth_token=token
)

print("pyannote OK")
PY


echo "Provision finished"
