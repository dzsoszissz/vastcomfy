#!/bin/bash
set -Eeuo pipefail

log() { echo "[my-provision] $*"; }
fail() { echo "[my-provision][FAIL] $*" >&2; exit 1; }

export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-64}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"
export HF_HUB_DISABLE_IMPLICIT_TOKEN="${HF_HUB_DISABLE_IMPLICIT_TOKEN:-0}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export DO_NOT_TRACK="${DO_NOT_TRACK:-1}"
LOG_FILE="${PROVISIONING_LOG:-/workspace/provisioning.log}"
exec > >(tee -a "$LOG_FILE") 2>&1
supervisecrl stop llama
hf download dzsoszmissz/Huihui-Qwen3.6-35B-A3B-abliterated-bet-GGUF --include "*Q6_K*" 
supervisecrl start llama
log "done"
