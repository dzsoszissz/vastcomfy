#!/bin/bash
set -Eeuo pipefail

log() { echo "[ltx23-msr] $*"; }
fail() { echo "[ltx23-msr][FAIL] $*" >&2; exit 1; }

: "${HF_TOKEN:?HF_TOKEN is required}"
case "$HF_TOKEN" in hf_*) ;; *) fail "HF_TOKEN must start with hf_" ;; esac

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/.cache/huggingface/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-/workspace/.cache/huggingface/xet}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-/workspace/.cache/huggingface/assets}"
export HF_XET_HIGH_PERFORMANCE="1"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="64"
export HF_HUB_DOWNLOAD_TIMEOUT="120"
export HF_HUB_ETAG_TIMEOUT="30"
export HF_HUB_DISABLE_XET="0"
export DO_NOT_TRACK="1"

COMFYUI_ROOT="${COMFYUI_ROOT:-/workspace/ComfyUI}"
CKPT_DIR="$COMFYUI_ROOT/models/checkpoints"
LORA_DIR="$COMFYUI_ROOT/models/loras"
VAE_DIR="$COMFYUI_ROOT/models/vae"
DIFFM_DIR="$COMFYUI_ROOT/models/diffusion_models"
TEXT_DIR="$COMFYUI_ROOT/models/text_encoders"
UPSCALE_DIR="$COMFYUI_ROOT/models/latent_upscale_models"
WORKFLOW_DIR="$COMFYUI_ROOT/user/default/workflows"
CUSTOM_DIR="$COMFYUI_ROOT/custom_nodes"
STAGE_ROOT="/workspace/.ltx23_msr_stage"
LOG_FILE="${PROVISIONING_LOG:-/workspace/provisioning_ltx23_msr.log}"

exec > >(tee -a "$LOG_FILE") 2>&1
[ -d "$COMFYUI_ROOT" ] || fail "ComfyUI root not found: $COMFYUI_ROOT"
mkdir -p "$CKPT_DIR" "$LORA_DIR" "$TEXT_DIR" "$UPSCALE_DIR" "$WORKFLOW_DIR" "$CUSTOM_DIR" "$STAGE_ROOT"

PY="/venv/main/bin/python3"
[ -x "$PY" ] || PY="$(command -v python3)"
PIP="$PY -m pip"

log "install/update huggingface_hub & hf_xet"
$PIP install -U --no-cache-dir "huggingface_hub[hf_xet]" >/tmp/msr_pip.log 2>&1 || fail "pip install failed"

# --- KÖTELEZŐ KORNIA VERZIÓ (Módosítás vége) ---
#pip install kornia==0.7.2 --upgrade --force-reinstall --no-cache-dir
#2. Függőségek: CSAK a szükségesek, verzióra kötve (No more conflicts!)
log "Cleaning environment and installing strict dependencies"
$PIP uninstall -y kornia kornia-rs || true
$PIP install  --upgrade --force-reinstall --no-cache-dir kornia==0.7.2 kornia-rs==0.1.14 torch torchvision torchaudio

HFCLI="$(dirname "$PY")/hf"
[ -x "$HFCLI" ] || HFCLI="$(command -v hf || command -v huggingface-cli)"
[ -x "$HFCLI" ] || fail "hf cli not found"

hf_file() {
  local repo="$1" src="$2" dest="$3"
  if [ -s "$dest" ]; then log "exists: $dest"; return 0; fi
  mkdir -p "$(dirname "$dest")"
  log "download: $repo :: $src -> $dest"
  log   "$HFCLI" download "$repo" "$src" --repo-type model --token "$HF_TOKEN" --local-dir "$(dirname "$dest")"
  "$HFCLI" download "$repo" "$src" --repo-type model --token "$HF_TOKEN" --local-dir "$(dirname "$dest")" >/tmp/msr_hf.log 2>&1
  [ -s "$(dirname "$dest")" ] || fail "download failed: $dest"
}

# 1. Custom nodes (MSR plugin + stabil alkatrészek)
log "installing custom nodes (MSR focused)"
for repo in \
    https://github.com/kijai/ComfyUI-KJNodes.git \
    https://github.com/Lightricks/ComfyUI-LTXVideo.git \
    https://github.com/liconstudio/ComfyUI-Licon-MSR.git; do
  name=$(basename "$repo" .git)
  [ -d "$CUSTOM_DIR/$name" ] || git clone --depth=1 "$repo" "$CUSTOM_DIR/$name"
  if [ -f "$CUSTOM_DIR/$name/requirements.txt" ]; then
    log "install requirements: $CUSTOM_DIR/$name/requirements.txt"
    $PIP install -r "$CUSTOM_DIR/$name/requirements.txt" >/tmp/ltx23_node_req.log 2>&1 || { cat /tmp/ltx23_node_req.log >&2; fail "node requirements install failed: $dest"; }
  fi
done

# 2. Requirements & Kornia patch (kizárólag LTXVideo-ra fókuszál)
log "installing requirements & patching kornia"
for dir in KJNodes LTXVideo; do
  log $PIP install -r "$CUSTOM_DIR/$dir/requirements.txt" >/dev/null 2>&1
  [ -f "$CUSTOM_DIR/$dir/requirements.txt" ] && $PIP install -r "$CUSTOM_DIR/$dir/requirements.txt" >/dev/null 2>&1
done
log "start python"
python3 << 'PYEOF'
import os
target = f"/workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/pyramid_blending.py"
c = open(target).read()
old = "from kornia.geometry.transform.pyramid import (\n    PyrUp,\n    build_laplacian_pyramid,\n    build_pyramid,\n    find_next_powerof_two,\n    is_powerof_two,\n    pad,\n)"
new = "from kornia.geometry.transform.pyramid import (\n    PyrUp,\n    build_laplacian_pyramid,\n    build_pyramid,\n    find_next_powerof_two,\n    is_powerof_two,\n)\nfrom torch.nn.functional import pad"
if old in c: open(target, 'w').write(c.replace(old, new))
PYEOF

# 3. Modellek (LTX 2.3 alap, distilled, gemma prompt, ingredients, upscalerok)
log "downloading models (MSR compatible)"
#hf_file "SulphurAI/Sulphur-2-base" "sulphur_dev_fp8mixed.safetensors" "$CKPT_DIR/ltx-2.3-22b-dev-fp8.safetensors"
#hf_file "szwagros/ltx-2.3-22b-distilled-1.1-bf16-fp8" "ltx-2.3-22b-distilled-1.1_transformer_fp8.safetensors" "$CKPT_DIR/ltx-2.3-22b-distilled-fp8.safetensors"
hf_file "Lightricks/LTX-2.3-fp8" "ltx-2.3-22b-distilled-fp8.safetensors" "$CKPT_DIR/ltx-2.3-22b-distilled-fp8.safetensors"
hf_file "coolthor/Sulphur-2-distilled-NVFP4" "ltx_full_vae.safetensors" "$VAE_DIR/ltx_full_vae.safetensors"
hf_file "coolthor/Sulphur-2-distilled-NVFP4" "sulphur_distil_nvfp4.safetensors" "$DIFFM_DIR/sulphur_distil_nvfp4.safetensors"
hf_file "Comfy-Org/ltx-2.3" "split_files/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors" "$LORA_DIR/distilled.safetensors"
#hf_file "Comfy-Org/ltx-2" "split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors" "$LORA_DIR/gemma_prompt.safetensors"
hf_file "LiconStudio/LTX-2.3-Multiple-Subject-Reference" "LTX-2.3-Licon-MSR-V1.safetensors" "$LORA_DIR/LTX-2.3"
hf_file "Comfy-Org/ltx-2" "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors" "$TEXT_DIR/gemma_encoder.fp4.mixed.safetensors"
#hf_file "Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients" "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors" "$LORA_DIR/ingredients.safetensors"
hf_file "Lightricks/LTX-2.3" "ltx-2.3-spatial-upscaler-x2-1.1.safetensors" "$UPSCALE_DIR/spatial.x2.11.safetensors"
hf_file "Lightricks/LTX-2.3" "ltx-2.3-temporal-upscaler-x2-1.0.safetensors" "$UPSCALE_DIR/temporal.x2.10.safetensors"

# 4. Workflow sablonok (csak JSON vázlatok, a logikát a plugin kezeli)
log "downloading workflow templates"
for wf in video_ltx2_3_t2v.json video_ltx2_3_i2v.json "First-Last-Frame to Video (LTX-2.3).json"; do
  curl -sLO --retry 3 https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/"$wf" || true
done

curl -s https://raw.githubusercontent.com/dzsoszissz/vastcomfy/refs/heads/main/LTX-2.3_MSR_sample_workflow_V1_working.json -o /workspace/ComfyUI/user/default/workflows/LTX-2.3_MSR_sample_workflow_V1_working.json
# 5. Manifest & indítás
cat > /workspace/ltx23_msr_ready.json << EOF
{
  "status": "ready",
  "model": "LiconStudio/LTX-2.3-Multiple-Subject-Reference",
  "plugin": "ComfyUI-Licon-MSR",
  "log": "$LOG_FILE"
}
EOF

if command -v supervisorctl >/dev/null 2>&1; then
  supervisorctl restart comfyui || true
  supervisorctl restart api-wrapper || true
fi
log "done. Old spec removed, MSR pipeline active."
