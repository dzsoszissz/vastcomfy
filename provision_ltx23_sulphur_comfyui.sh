#!/bin/bash
set -Eeuo pipefail

log() { echo "[ltx23-provision] $*"; }
fail() { echo "[ltx23-provision][FAIL] $*" >&2; exit 1; }

: "${HF_TOKEN:?HF_TOKEN is required}"
case "$HF_TOKEN" in hf_*) ;; *) fail "HF_TOKEN must start with hf_" ;; esac

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
CUSTOM_DIR="$COMFYUI_ROOT/custom_nodes"
STAGE_ROOT="/workspace/.ltx23_stage"
LOG_FILE="${PROVISIONING_LOG:-/workspace/provisioning_ltx23.log}"

exec > >(tee -a "$LOG_FILE") 2>&1

[ -d "$COMFYUI_ROOT" ] || fail "ComfyUI root not found: $COMFYUI_ROOT"
mkdir -p "$CKPT_DIR" "$LORA_DIR" "$TEXT_DIR" "$UPSCALE_DIR" "$WORKFLOW_DIR" "$INPUT_DIR" "$OUTPUT_DIR" "$TEMP_DIR" "$CUSTOM_DIR" "$STAGE_ROOT" "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$HF_ASSETS_CACHE"

FREE_GB="$(df -BG /workspace | awk 'NR==2 {gsub(/G/,"",$4); print $4}')"
[ "${FREE_GB:-0}" -ge 40 ] || fail "free disk below 60GB on /workspace: ${FREE_GB:-0}GB"

PY="/venv/main/bin/python"
[ -x "$PY" ] || PY="/opt/venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 || true)"
[ -n "$PY" ] || fail "python3 not found"

PIP="$PY -m pip"
log "install/update huggingface_hub hf_xet"
$PIP install -U --no-cache-dir "huggingface_hub[hf_xet]" >/tmp/ltx23_pip_hf.log 2>&1 || { cat /tmp/ltx23_pip_hf.log >&2; fail "pip install huggingface_hub[hf_xet] failed"; }

HFCLI="$(dirname "$PY")/hf"
[ -x "$HFCLI" ] || HFCLI="$(command -v hf || true)"
[ -x "$HFCLI" ] || HFCLI="$(command -v huggingface-cli || true)"
[ -x "$HFCLI" ] || fail "hf cli not found"

hf_file() {
  local repo="$1" src="$2" dest="$3"
  local stage src_path
  stage="$STAGE_ROOT/$(echo "$repo/$src" | tr '/:' '__')"
  src_path="$stage/$src"
  if [ -s "$dest" ]; then
    log "exists: $dest"
    return 0
  fi
  rm -rf "$stage"
  mkdir -p "$stage" "$(dirname "$dest")"
  log "download: $repo :: $src"
  "$HFCLI" download "$repo" "$src" --repo-type model --token "$HF_TOKEN" --local-dir "$stage" >/tmp/ltx23_hf.log 2>&1 || { cat /tmp/ltx23_hf.log >&2; fail "hf download failed: $repo :: $src"; }
  [ -s "$src_path" ] || fail "downloaded file missing: $src_path"
  mv -f "$src_path" "$dest"
  rm -rf "$stage"
  [ -s "$dest" ] || fail "target missing after move: $dest"
}

git_install_node() {
  local repo_url="$1" dest="$2"
  if [ -d "$dest/.git" ]; then
    log "custom node exists: $dest"
  else
    rm -rf "$dest"
    log "clone custom node: $repo_url"
    git clone --depth 1 "$repo_url" "$dest" || fail "git clone failed: $repo_url"
  fi
  if [ -f "$dest/requirements.txt" ]; then
    log "install requirements: $dest/requirements.txt"
    $PIP install -r "$dest/requirements.txt" >/tmp/ltx23_node_req.log 2>&1 || { cat /tmp/ltx23_node_req.log >&2; fail "node requirements install failed: $dest"; }
  fi
}

curl_file() {
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then
    log "exists: $dest"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  log "download: $url"
  curl -fL --retry 5 --retry-delay 5 --connect-timeout 30 --max-time 600 "$url" -o "$dest.tmp" || fail "curl download failed: $url"
  [ -s "$dest.tmp" ] || fail "downloaded temp file is empty: $dest.tmp"
  mv -f "$dest.tmp" "$dest"
  [ -s "$dest" ] || fail "target missing after curl: $dest"
}

log "install ComfyUI-LTXVideo custom nodes"
git_install_node "https://github.com/Lightricks/ComfyUI-LTXVideo.git" "$CUSTOM_DIR/ComfyUI-LTXVideo"

log "download models"
hf_file "SulphurAI/Sulphur-2-base" "sulphur_dev_fp8mixed.safetensors" "$CKPT_DIR/ltx-2.3-22b-dev-fp8.safetensors"
hf_file "Comfy-Org/ltx-2.3" "split_files/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors" "$LORA_DIR/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
hf_file "Comfy-Org/ltx-2" "split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors" "$LORA_DIR/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors"
hf_file "Comfy-Org/ltx-2" "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors" "$TEXT_DIR/gemma_3_12B_it_fp4_mixed.safetensors"
hf_file "Lightricks/LTX-2.3" "ltx-2.3-spatial-upscaler-x2-1.1.safetensors" "$UPSCALE_DIR/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
hf_file "Lightricks/LTX-2.3" "ltx-2.3-temporal-upscaler-x2-1.0.safetensors" "$UPSCALE_DIR/ltx-2.3-temporal-upscaler-x2-1.0.safetensors"

log "download workflows"
curl_file "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_ltx2_3_i2v.json" "$WORKFLOW_DIR/video_ltx2_3_i2v.json"
curl_file "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_ltx2_3_t2v.json" "$WORKFLOW_DIR/video_ltx2_3_t2v.json"
curl_file "https://raw.githubusercontent.com/Comfy-Org/Subgraph-Blueprints/main/First-Last-Frame%20to%20Video%20%28LTX-2.3%29.json" "$WORKFLOW_DIR/First-Last-Frame to Video (LTX-2.3).json"

log "validate files"
for f in \
  "$CKPT_DIR/ltx-2.3-22b-dev-fp8.safetensors" \
  "$LORA_DIR/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors" \
  "$LORA_DIR/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors" \
  "$TEXT_DIR/gemma_3_12B_it_fp4_mixed.safetensors" \
  "$UPSCALE_DIR/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
  "$UPSCALE_DIR/ltx-2.3-temporal-upscaler-x2-1.0.safetensors" \
  "$WORKFLOW_DIR/video_ltx2_3_i2v.json" \
  "$WORKFLOW_DIR/video_ltx2_3_t2v.json" \
  "$WORKFLOW_DIR/First-Last-Frame to Video (LTX-2.3).json"; do
  [ -s "$f" ] || fail "missing/empty: $f"
done

cat > /workspace/ltx23_ready.json <<MANIFEST
{
  "status": "ready",
  "checkpoint": "$CKPT_DIR/ltx-2.3-22b-dev-fp8.safetensors",
  "lora_distilled": "$LORA_DIR/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors",
  "lora_prompt": "$LORA_DIR/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
  "text_encoder": "$TEXT_DIR/gemma_3_12B_it_fp4_mixed.safetensors",
  "spatial_upscaler": "$UPSCALE_DIR/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
  "temporal_upscaler": "$UPSCALE_DIR/ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
  "workflow_i2v": "$WORKFLOW_DIR/video_ltx2_3_i2v.json",
  "workflow_t2v": "$WORKFLOW_DIR/video_ltx2_3_t2v.json",
  "workflow_flf": "$WORKFLOW_DIR/First-Last-Frame to Video (LTX-2.3).json",
  "log": "$LOG_FILE"
}
MANIFEST
COMFY_DIR="${COMFY_DIR:-/workspace/ComfyUI}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$COMFY_DIR"
mkdir -p custom_nodes
cd custom_nodes

if [ ! -d ComfyUI-KJNodes ]; then
  git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes.git
else
  cd ComfyUI-KJNodes
  git pull --ff-only
  cd ..
fi

if [ ! -d ComfyUI-LTXVideo ]; then
  git clone --depth=1 https://github.com/Lightricks/ComfyUI-LTXVideo.git
else
  cd ComfyUI-LTXVideo
  git pull --ff-only
  cd ..
fi

if [ -f ComfyUI-KJNodes/requirements.txt ]; then
  "$PYTHON_BIN" -m pip install -r ComfyUI-KJNodes/requirements.txt
fi
if [ -f ComfyUI-LTXVideo/requirements.txt ]; then
  "$PYTHON_BIN" -m pip install -r ComfyUI-LTXVideo/requirements.txt
fi

log "restart comfyui if supervisor program exists"
if command -v supervisorctl >/dev/null 2>&1; then
  supervisorctl status | awk '{print $1}' | grep -Ei 'comfy|comfyui' | while read -r svc; do supervisorctl restart "$svc" || true; done
fi

log "done"
