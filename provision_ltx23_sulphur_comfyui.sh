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
DIFF_DIR="$COMFYUI_ROOT/models/diffusion_models"

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
$PIP install "jiwer<3.0"
# --- KÖTELEZŐ KORNIA VERZIÓ (Módosítás vége) ---
#pip install kornia==0.7.2 --upgrade --force-reinstall --no-cache-dir
#2. Függőségek: CSAK a szükségesek, verzióra kötve (No more conflicts!)
log "Cleaning environment and installing strict dependencies"
$PIP uninstall -y kornia kornia-rs || true
# A torch/torchvision/torchaudio-t NEM telepitjuk ujra: a szerver drivereh
# ez illo, mar mukodo torch marad. A kornia --no-deps, hogy ne huzzon torch-ot.
$PIP install --no-cache-dir --force-reinstall --no-deps kornia==0.7.2 kornia-rs==0.1.14

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
# https://github.com/Saganaki22/ComfyUI-KugelAudio.git \
log "installing custom nodes (MSR focused)"
# Custom node repók feladatonként csoportosítva:
#   KÖZÖS/SEGÉD: KJNodes, KayTool
#   KÉP (Flux/Chroma): ControlAltAI-Nodes
#   VIDEÓ (LTX-2.3): ComfyUI-LTXVideo, ComfyUI-Licon-MSR
#   AUDIO (F5-TTS): F5-TTS-ComfyUI
for repo in \
    https://github.com/Lightricks/ComfyUI-LTXVideo.git \
    https://github.com/liconstudio/ComfyUI-Licon-MSR.git \
    https://github.com/niknah/ComfyUI-F5-TTS.git; do
  name=$(basename "$repo" .git)
  [ -d "$CUSTOM_DIR/$name" ] || git clone --depth=1 "$repo" "$CUSTOM_DIR/$name"
  if [ -f "$CUSTOM_DIR/$name/requirements.txt" ]; then
    log "install requirements (torch-erzekeny csomagok kiszurve): $CUSTOM_DIR/$name/requirements.txt"
    # A torch-ot ujratelepito/ranto csomagokat kiszurjuk, hogy a mukodo torch maradjon.
    grep -viE '^(torch|torchvision|torchaudio|torchcodec|torch-time-stretch|torch_time_stretch)([<>=!~ ]|$)' \
      "$CUSTOM_DIR/$name/requirements.txt" > "$CUSTOM_DIR/$name/requirements.filtered.txt" || true
    $PIP install --no-cache-dir -r "$CUSTOM_DIR/$name/requirements.filtered.txt" >/tmp/ltx23_node_req.log 2>&1 || { cat /tmp/ltx23_node_req.log >&2; fail "node requirements install failed"; }
  fi
done

# 2. Requirements & Kornia patch (kizárólag LTXVideo-ra fókuszál)
log "installing requirements & patching kornia"
for dir in LTXVideo; do
  if [ -f "$CUSTOM_DIR/$dir/requirements.txt" ]; then
    grep -viE '^(torch|torchvision|torchaudio|torchcodec|torch-time-stretch|torch_time_stretch)([<>=!~ ]|$)' \
      "$CUSTOM_DIR/$dir/requirements.txt" > "$CUSTOM_DIR/$dir/requirements.filtered.txt" || true
    $PIP install --no-cache-dir -r "$CUSTOM_DIR/$dir/requirements.filtered.txt" >/dev/null 2>&1
  fi
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

# ===== VIDEÓ modellek (LTX-2.3 / Sulphur / Gemma / upscaler) =====
#hf_file "SulphurAI/Sulphur-2-base" "sulphur_dev_fp8mixed.safetensors" "$CKPT_DIR/ltx-2.3-22b-dev-fp8.safetensors"
#hf_file "szwagros/ltx-2.3-22b-distilled-1.1-bf16-fp8" "ltx-2.3-22b-distilled-1.1_transformer_fp8.safetensors" "$CKPT_DIR/ltx-2.3-22b-distilled-fp8.safetensors"
#hf_file "Lightricks/LTX-2.3-fp8" "ltx-2.3-22b-distilled-fp8.safetensors" "$CKPT_DIR/ltx-2.3-22b-distilled-fp8.safetensors"
#hf_file "coolthor/Sulphur-2-distilled-NVFP4" "ltx_full_vae.safetensors" "$VAE_DIR/ltx_full_vae.safetensors"
#hf_file "coolthor/Sulphur-2-distilled-NVFP4" "sulphur_distil_nvfp4.safetensors" "$CKPT_DIR/sulphur_distil_nvfp4.safetensors"
hf_file "Winnougan/Sulphur-2-LTX-2.3" "sulphur_distill_fp8.safetensors" "$CKPT_DIR/sulphur_distill_fp8.safetensors"
#hf_file "Lightricks/LTX-2.3" "ltx-2.3-22b-distilled-1.1.safetensors" "$CKPT_DIR/ltx-2.3-22b-distilled-1.1.safetensors"
#hf_file "Comfy-Org/ltx-2.3" "split_files/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors" "$LORA_DIR/distilled.safetensors"
#hf_file "Comfy-Org/ltx-2" "split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors" "$LORA_DIR/gemma_prompt.safetensors"
hf_file "LiconStudio/LTX-2.3-Multiple-Subject-Reference" "LTX-2.3-Licon-MSR-V1.safetensors" "$LORA_DIR/LTX-2.3"
#hf_file "Comfy-Org/ltx-2" "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors" "$TEXT_DIR/gemma_encoder.fp4.mixed.safetensors"
hf_file "Sikaworld1990/gemma-3-12b-qat-abliterated-sikaworld-fp4-ltx2" "Gemma3-12B-NVFP4-Sikaworld-Pure.safetensors" "$TEXT_DIR/Gemma3-12B-NVFP4-Sikaworld-Pure.safetensors"
#hf_file "Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients" "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors" "$LORA_DIR/ingredients.safetensors"
hf_file "Lightricks/LTX-2.3" "ltx-2.3-spatial-upscaler-x2-1.1.safetensors" "$UPSCALE_DIR/spatial.x2.11.safetensors"
hf_file "Lightricks/LTX-2.3" "ltx-2.3-temporal-upscaler-x2-1.0.safetensors" "$UPSCALE_DIR/temporal.x2.10.safetensors"

# ===== AUDIO modellek (F5-TTS magyar) =====
# niknah node: checkpoints/F5-TTS/, a vocab a modell nevén .txt kiterjesztéssel
F5_DIR="$CKPT_DIR/F5-TTS"
hf_file "sarpba/F5-TTS_V1_hun_v2" "model_927900.safetensors" "$F5_DIR/model_927900.safetensors"
hf_file "sarpba/F5-TTS_V1_hun_v2" "vocab.txt" "$F5_DIR/vocab.txt"
cp "$F5_DIR/vocab.txt" $F5_DIR/model_927900.txt"
# ===== KÉP modellek (Chroma/UnCanny + Qwen-Edit + Kontext) =====
# [ELTÁVOLÍTVA - felesleges, Qwen kép-wf nem hasznalja] hf_file "mingyi456/UnCanny-Photorealism-Chroma-DF11-ComfyUI" "uncannyPhotorealism_v12-DF11.safetensors" "$DIFF_DIR/uncannyPhotorealism_v12-DF11.safetensors"
# [ELTÁVOLÍTVA - felesleges, Qwen kép-wf nem hasznalja] hf_file "comfyanonymous/flux_text_encoders" "t5xxl_fp8_e4m3fn_scaled.safetensors" "$TEXT_DIR/t5xxl_fp8_e4m3fn_scaled.safetensors"
# [ELTÁVOLÍTVA - felesleges, Qwen kép-wf nem hasznalja] hf_file "lodestones/Chroma" "ae.safetensors" "$VAE_DIR/ae.safetensors"
hf_file "Phr00t/Qwen-Image-Edit-Rapid-AIO" "v23/Qwen-Rapid-AIO-NSFW-v23.safetensors" "$CKPT_DIR/Qwen-Rapid-AIO-NSFW-v23.safetensors"
# [ELTÁVOLÍTVA - felesleges, Qwen kép-wf nem hasznalja] hf_file "Comfy-Org/flux1-kontext-dev_ComfyUI" "split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors" "$DIFF_DIR/flux1-dev-kontext_fp8_scaled.safetensors"
# [ELTÁVOLÍTVA - felesleges, Qwen kép-wf nem hasznalja] hf_file "comfyanonymous/flux_text_encoders" "clip_l.safetensors" "$TEXT_DIR/clip_l.safetensors"
# 4. Workflow sablonok (csak JSON vázlatok, a logikát a plugin kezeli)
log "downloading workflow templates"
for wf in video_ltx2_3_t2v.json video_ltx2_3_i2v.json "First-Last-Frame to Video (LTX-2.3).json"; do
  curl -sLO --retry 3 https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/"$wf" || true
done

curl -s https://raw.githubusercontent.com/dzsoszissz/vastcomfy/refs/heads/main/LTX-2.3_MSR_sample_workflow_V1_working.json -o /workspace/ComfyUI/user/default/workflows/LTX-2.3_MSR_sample_workflow_V1_working.json
curl -s https://raw.githubusercontent.com/niknah/ComfyUI-F5-TTS/refs/heads/main/example_workflows/simple_ComfyUI_F5TTS_workflow.json -o /workspace/ComfyUI/user/default/workflows/F5TTS_hun_workflow.json
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
