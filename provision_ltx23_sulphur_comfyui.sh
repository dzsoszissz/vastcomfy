#!/bin/bash
set -Eeuo pipefail

log() { echo "[ltx23-provision] $*"; }
fail() { echo "[ltx23-provision][FAIL] $*" >&2; exit 1; }

: "${HF_TOKEN:?HF_TOKEN is required}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/.cache/huggingface/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-/workspace/.cache/huggingface/xet}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-/workspace/.cache/huggingface/assets}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-64}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"
export HF_HUB_DISABLE_IMPLICIT_TOKEN="${HF_HUB_DISABLE_IMPLICIT_TOKEN:-0}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export DO_NOT_TRACK="${DO_NOT_TRACK:-1}"

COMFYUI_ROOT="${COMFYUI_ROOT:-/workspace/ComfyUI}"
CKPT_DIR="${COMFYUI_CHECKPOINT_DIR:-$COMFYUI_ROOT/models/checkpoints}"
LORA_DIR="${COMFYUI_LORA_DIR:-$COMFYUI_ROOT/models/loras}"
TEXT_DIR="${COMFYUI_TEXT_ENCODER_DIR:-$COMFYUI_ROOT/models/text_encoders}"
UPSCALE_DIR="${COMFYUI_LATENT_UPSCALE_DIR:-$COMFYUI_ROOT/models/latent_upscale_models}"
WORKFLOW_DIR="${COMFYUI_WORKFLOW_DIR:-$COMFYUI_ROOT/user/default/workflows}"
INPUT_DIR="${COMFYUI_INPUT_DIR:-$COMFYUI_ROOT/input}"
OUTPUT_DIR="${COMFYUI_OUTPUT_DIR:-$COMFYUI_ROOT/output}"
TEMP_DIR="${COMFYUI_TEMP_DIR:-$COMFYUI_ROOT/temp}"
TMP_DIR="/workspace/.ltx23_provision_tmp"

mkdir -p "$CKPT_DIR" "$LORA_DIR" "$TEXT_DIR" "$UPSCALE_DIR" "$WORKFLOW_DIR" "$INPUT_DIR" "$OUTPUT_DIR" "$TEMP_DIR" "$TMP_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$HF_ASSETS_CACHE"

PY="/venv/main/bin/python"
if [ ! -x "$PY" ]; then PY="$(command -v python3 || true)"; fi
[ -n "$PY" ] || fail "python3 not found"

log "install huggingface_hub + hf_xet"
"$PY" -m pip install -U --no-cache-dir "huggingface_hub[hf_xet]" >/tmp/ltx23_pip.log 2>&1 || { cat /tmp/ltx23_pip.log >&2; fail "pip install failed"; }

if command -v hf >/dev/null 2>&1; then
  HFCLI="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HFCLI="huggingface-cli"
else
  fail "hf cli not found after install"
fi

hf_file() {
  local repo="$1" file="$2" dest="$3"
  local dest_dir
  dest_dir="$(dirname "$dest")"
  mkdir -p "$dest_dir"
  if [ -s "$dest" ]; then
    log "exists: $dest"
    return 0
  fi
  rm -rf "$TMP_DIR/download"
  mkdir -p "$TMP_DIR/download"
  log "download: $repo :: $file"
  "$HFCLI" download "$repo" "$file" --repo-type model --token "$HF_TOKEN" --local-dir "$TMP_DIR/download" --local-dir-use-symlinks False >/tmp/ltx23_hf.log 2>&1 || { cat /tmp/ltx23_hf.log >&2; fail "hf download failed: $repo :: $file"; }
  [ -s "$TMP_DIR/download/$file" ] || fail "downloaded file missing: $TMP_DIR/download/$file"
  cp -f "$TMP_DIR/download/$file" "$dest"
  [ -s "$dest" ] || fail "copy failed: $dest"
}

hf_file "Seregil13th/Sulphur-2-base" "sulphur_dev_fp8mixed.safetensors" "$CKPT_DIR/sulphur_dev_fp8mixed.safetensors"
ln -sf "$CKPT_DIR/sulphur_dev_fp8mixed.safetensors" "$CKPT_DIR/ltx-2.3-22b-dev-fp8.safetensors"

hf_file "Seregil13th/Sulphur-2-base" "distill_loras/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors" "$LORA_DIR/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors"
ln -sf "$LORA_DIR/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors" "$LORA_DIR/ltx-2.3-22b-distilled-lora-384.safetensors"

hf_file "Comfy-Org/ltx-2" "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors" "$TEXT_DIR/gemma_3_12B_it_fp4_mixed.safetensors"
hf_file "Lightricks/LTX-2.3" "ltx-2.3-spatial-upscaler-x2-1.0.safetensors" "$UPSCALE_DIR/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

log "download FLF workflow"
WF_URL="https://raw.githubusercontent.com/Comfy-Org/Subgraph-Blueprints/main/First-Last-Frame%20to%20Video%20%28LTX-2.3%29.json"
WF_DEST="$WORKFLOW_DIR/First-Last-Frame to Video (LTX-2.3).json"
if [ ! -s "$WF_DEST" ]; then
  curl -fL --retry 5 --retry-delay 5 "$WF_URL" -o "$WF_DEST" || fail "workflow download failed"
fi
[ -s "$WF_DEST" ] || fail "workflow missing: $WF_DEST"

log "write manifest"
cat > /workspace/ltx23_ready.json <<MANIFEST
{
  "status": "ready",
  "checkpoint": "$CKPT_DIR/sulphur_dev_fp8mixed.safetensors",
  "checkpoint_alias": "$CKPT_DIR/ltx-2.3-22b-dev-fp8.safetensors",
  "lora": "$LORA_DIR/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors",
  "lora_alias": "$LORA_DIR/ltx-2.3-22b-distilled-lora-384.safetensors",
  "text_encoder": "$TEXT_DIR/gemma_3_12B_it_fp4_mixed.safetensors",
  "latent_upscaler": "$UPSCALE_DIR/ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
  "workflow": "$WF_DEST"
}
MANIFEST

log "restart comfyui if supervisor program exists"
if command -v supervisorctl >/dev/null 2>&1; then
  supervisorctl status | awk '{print $1}' | grep -Ei 'comfy|comfyui' | while read -r svc; do supervisorctl restart "$svc" || true; done
fi

log "done"
