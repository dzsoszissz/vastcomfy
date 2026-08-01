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
# FONTOS VRAM-megfontolás: a NLLB/Qwen3-Stylizer/F5-TTS modelleket
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

cat > "${REPO_DIR}/backend/routers/custom_ai/router.py" <<'PYEOF'
"""
custom_ai/router.py — sajat router: forditas (NLLB-200 + Qwen3-4B-Instruct-2507)
es magyar hangklonozas (F5-TTS). A backend/main.py koti be a meglevo
WhisperX-WebUI FastAPI-alkalmazasba.

##############################################################################
##  FIGYELEM — LICENC KORLATOZAS!                                          ##
##  A /tts vegpont a Maxdorger29/f5-tts-hungarian CC-BY-NC-4.0 licencu     ##
##  sulyait hasznalja -> CSAK NEM KERESKEDELMI CELRA HASZNALHATO!          ##
##  Lasd a provisioning szkript vonatkozo lepesenek kommentjeit.           ##
##############################################################################
"""
import functools
import io
import json
import os
import re
import tempfile
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

TRANSLATION_MODEL_DIR = os.environ.get("TRANSLATION_MODEL_DIR", "/workspace/models/Translation")
F5TTS_HU_DIR = os.environ.get("F5TTS_HU_DIR", "/workspace/models/F5-TTS-Hungarian")

custom_ai_router = APIRouter(prefix="/custom-ai", tags=["Custom AI (Translate + TTS)"])


# ============================================================
# Forditas: NLLB-200 1.3B (nyers forditas) -> Qwen3-4B-Instruct-2507
# (magyar stilizalas)
# ============================================================

class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "eng_Latn"  # FLORES-200 nyelvkod
    target_lang: str = "hun_Latn"
    stylize: bool = True


class TranslateResponse(BaseModel):
    raw_translation: str
    stylized_translation: str


@functools.lru_cache
def get_nllb():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B", cache_dir=TRANSLATION_MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-1.3B", cache_dir=TRANSLATION_MODEL_DIR, torch_dtype=torch.float16
    ).to("cuda")
    return tokenizer, model


@functools.lru_cache
def get_stylizer_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", cache_dir=TRANSLATION_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507",
        cache_dir=TRANSLATION_MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    return tokenizer, model


def nllb_translate(text: str, source_lang: str, target_lang: str) -> str:
    import torch

    tokenizer, model = get_nllb()
    tokenizer.src_lang = source_lang
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    target_id = tokenizer.convert_tokens_to_ids(target_lang)
    with torch.no_grad():
        generated = model.generate(**inputs, forced_bos_token_id=target_id, max_new_tokens=512)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def stylize_translation(raw_text: str, original_text: str) -> str:
    import torch

    tokenizer, model = get_stylizer_model()
    # A korabban hasznalt Racka-4B (magyar-hangolt, de dual-mode Qwen3-4B
    # alapu) elesben megbizhatatlannak bizonyult: idonkent visszaidezte a
    # sajat rendszer-promptjat, duplikalta a valaszt, vagy beleirta a
    # sablon-cimkeket a kimenetbe. A Qwen3-4B-Instruct-2507 ennek pont az
    # ellentetje: DEDIKALT, csak non-thinking modu valtozat (nincs is
    # <think> blokk generalasi kepessege architekturalisan), amit
    # kifejezetten az instrukciokovetes es a tobbnyelvu illeszkedes
    # javitasara frissitettek (2025 julius). A "VEGSO:" jelolo + a
    # kinyeres-alapu feldolgozas ennek ellenere megmarad tovabbi
    # biztonsagi halonak — barmely modellnel hasznos, nem csak a regi
    # Rackanal.
    messages = [
        {
            "role": "system",
            "content": (
                "Te egy profi magyar szinkron- es feliratforditó vagy. A feladatod, hogy egy "
                "nyers, gépi fordítást természetes, folyékony, köznyelvi magyarra igazíts, "
                "az eredeti jelentés megtartása mellett."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Eredeti szöveg: {original_text}\n"
                f"Nyers fordítás: {raw_text}\n\n"
                "Valaszolj PONTOSAN ebben a formatumban, semmi mast ne irj:\n"
                "VEGSO: <a javított magyar mondat, egyetlen sorban>"
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            # repetition_penalty + no_repeat_ngram_size: tovabbi vedelem
            # barmilyen degeneralt ismetles ellen (pl. korabban megfigyelt
            # "Tess. Tess." tipusu duplikacio) — barmely modellnel hasznos
            # biztonsagi halo, nem csak a regi Rackanal jelentkezo hibara
            # valo reakciokent.
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
    output_ids = generated[0][inputs["input_ids"].shape[1]:]
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    # Vedekezo tisztitas: a Qwen3-4B-Instruct-2507 architekturalisan nem
    # general <think> blokkot, de ha valamiert megis maradna (pl. jovobeli
    # modellcsere), vagjuk le, mielott a jelolot keresnenk.
    output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()

    # A "VEGSO:" jelolo utani, ELSO sor a tenyleges valasz — fuggetlenul
    # attol, mi elozi meg (ismetles, visszaidezett prompt-reszlet, stb.).
    match = re.search(r"VEGSO:\s*(.+)", output, flags=re.IGNORECASE)
    if match:
        output = match.group(1).strip().split("\n")[0].strip()
    else:
        output = ""

    # Ha a jelolo nem talalhato, LEZARATLAN "<think>" maradt, vagy a
    # kinyert valasz ures — inkabb a nyers NLLB-forditast adjuk vissza,
    # mint hasznalhatatlan/megbizhatatlan szoveget.
    if not output or "<think>" in output:
        return raw_text

    return output


@custom_ai_router.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    try:
        raw = nllb_translate(req.text, req.source_lang, req.target_lang)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"NLLB fordítási hiba: {exc}")

    if not req.stylize:
        return TranslateResponse(raw_translation=raw, stylized_translation=raw)

    try:
        stylized = stylize_translation(raw, req.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Qwen3 stilizálási hiba: {exc}")

    return TranslateResponse(raw_translation=raw, stylized_translation=stylized)


# ============================================================
# Kotegelt, KONTEXTUS-TUDATOS forditas: a TELJES parbeszedet egyben
# kapja a Qwen3-4B-Instruct-2507 (NLLB nelkul, kozvetlenul), igy latja
# az osszefuggeseket es konzisztens stilust/hangnemet tud tartani
# beszelonkent — nem csak izolalt, egymastol fuggetlen mondatokat forgat.
# ============================================================

class BatchSegmentIn(BaseModel):
    id: int
    speaker: Optional[str] = None
    text: str


class BatchTranslateRequest(BaseModel):
    segments: List[BatchSegmentIn]


class BatchSegmentOut(BaseModel):
    id: int
    speaker: Optional[str] = None
    text_en: str
    text_hu: str


class BatchTranslateResponse(BaseModel):
    segments: List[BatchSegmentOut]


def translate_batch_with_context(segments: list) -> list:
    import torch

    tokenizer, model = get_stylizer_model()

    lines = []
    for seg in segments:
        speaker = seg.get("speaker") or "?"
        lines.append(
            '{"id": %d, "speaker": %s, "text": %s}'
            % (seg["id"], json.dumps(speaker, ensure_ascii=False), json.dumps(seg["text"], ensure_ascii=False))
        )
    segments_json = "[\n  " + ",\n  ".join(lines) + "\n]"

    messages = [
        {
            "role": "system",
            "content": (
                "Te egy profi magyar szinkron- es feliratforditó vagy. Egy teljes jelenet "
                "angol párbeszédét kapod, időrendi sorrendben, beszélő-címkékkel. Fordítsd le "
                "természetes, folyékony, köznyelvi magyarra, ÚGY, HOGY figyelembe veszed a "
                "teljes beszélgetés kontextusát, és konzisztens stílust/hangnemet tartasz meg "
                "beszélőnként az egész jeleneten át."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Párbeszéd (JSON tömb):\n{segments_json}\n\n"
                "Válaszolj PONTOSAN ebben a formátumban, más semmit ne írj — minden bemeneti "
                "id-hoz pontosan egy kimeneti elem tartozzon, ugyanabban a sorrendben:\n"
                "JSON_KEZDET\n"
                '[{"id": <id>, "text_hu": "<magyar fordítás>"}, ...]\n'
                "JSON_VEGE"
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # A kimenet meretenek dinamikus skalazasa a szegmensszammal — egy
    # nagyobb parbeszednek tobb token kell, hogy ne vagja le a valaszt.
    max_tokens = max(1024, len(segments) * 80)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
    output_ids = generated[0][inputs["input_ids"].shape[1]:]
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()

    match = re.search(r"JSON_KEZDET\s*(.+?)\s*JSON_VEGE", output, flags=re.DOTALL)
    if match:
        candidate = match.group(1).strip()
    else:
        # Vedekezo fallback: a jelolok hianyaban probaljuk megkeresni az
        # elso "[" es utolso "]" kozotti reszt.
        start = output.find("[")
        end = output.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("A modell valasza nem tartalmazott ertelmezheto JSON-t.")
        candidate = output[start : end + 1]

    # Gyakori LLM-JSON hiba: felesleges vesszo a listak/objektumok vegen —
    # ezt levagjuk, mielott parse-olnank.
    candidate = re.sub(r",\s*([\]}])", r"\1", candidate)

    parsed = json.loads(candidate)
    by_id = {item["id"]: item.get("text_hu", "") for item in parsed if "id" in item}

    result = []
    for seg in segments:
        result.append(
            {
                "id": seg["id"],
                "speaker": seg.get("speaker"),
                "text_en": seg["text"],
                "text_hu": by_id.get(seg["id"], ""),
            }
        )
    return result


@custom_ai_router.post("/translate_batch", response_model=BatchTranslateResponse)
def translate_batch(req: BatchTranslateRequest):
    try:
        segments_in = [s.dict() for s in req.segments]
        result = translate_batch_with_context(segments_in)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Kötegelt fordítási hiba: {exc}")

    return BatchTranslateResponse(segments=[BatchSegmentOut(**r) for r in result])


# ============================================================
# F5-TTS magyar hangklonozas — CC-BY-NC-4.0, CSAK NEM KERESKEDELMI CELRA!
# ============================================================

@functools.lru_cache
def get_f5tts():
    # A Maxdorger29/f5-tts-hungarian modellkartya sajat ajanlasa szerint:
    # torchaudio.load lecserelese soundfile-ra platformfuggetlen betoltes
    # miatt (ismert kompatibilitasi tuneteket kerul el).
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    def _patched_load(fp, **kw):
        data, sr = sf.read(str(fp), dtype="float32")
        if data.ndim == 1:
            data = data[np.newaxis, :]
        else:
            data = data.T
        return torch.from_numpy(data), sr

    torchaudio.load = _patched_load

    from f5_tts.api import F5TTS

    model = F5TTS(
        model="F5TTS_v1_Base",
        ckpt_file=os.path.join(F5TTS_HU_DIR, "model_last_final.safetensors"),
        vocab_file=os.path.join(F5TTS_HU_DIR, "vocab.txt"),
        device="cuda",
        use_ema=True,
    )
    return model


@custom_ai_router.post("/tts")
async def tts(
    ref_audio: UploadFile = File(..., description="Referencia hangminta (5-15 mp, pl. speaker_00.wav)"),
    ref_text: str = Form(..., description="A referencia hang pontos átirata"),
    gen_text: str = Form(..., description="A felolvasandó/generálandó magyar szöveg"),
    speed: float = Form(
        0.85,
        description=(
            "Az F5-TTS a kimenet hosszat a ref_text/gen_text KARAKTERARANYABOL "
            "becsuli — ha a ket szoveg kulonbozo nyelvu (nalunk tipikusan angol "
            "ref_text, magyar gen_text), ez az arany felreviheti a becslest es "
            "korai levagast okozhat. Az alapertelmezett 0.85 (a konyvtar sajat "
            "alapertelmezese: 1.0) tobb keretet/idot enged a generalasnak "
            "biztonsagi tartalekkent. Ha meg igy is levagna a vege, probalj meg "
            "kisebb erteket (pl. 0.7); ha tul lassunak/nyujtottnak hangzik, "
            "novelheted 1.0 fele."
        ),
    ),
):
    import soundfile as sf

    model = get_f5tts()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_ref:
        tmp_ref.write(await ref_audio.read())
        tmp_ref_path = tmp_ref.name

    try:
        wav, sr, _ = model.infer(ref_file=tmp_ref_path, ref_text=ref_text, gen_text=gen_text, speed=speed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"F5-TTS generálási hiba: {exc}")
    finally:
        os.unlink(tmp_ref_path)

    buffer = io.BytesIO()
    sf.write(buffer, wav, sr, format="WAV")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="audio/wav")


@custom_ai_router.get("/health")
def health():
    return {"status": "ok"}
PYEOF

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
echo "  whisperx-backend regisztrálva és elindítva a supervisor alatt (port 8000)"
echo "  napló: /var/log/portal/whisperx-backend.log"

###############################################################################
# 8. Fordítási modellek előtöltése (NLLB-200 1.3B nyers fordítás +
#    Qwen3-4B-Instruct-2507 magyar stilizálás)
#
# Ez a szakasz NEM része a WhisperX-WebUI repónak — külön, a jövőbeli saját
# fordítási szolgáltatáshoz készíti elő a modelleket, ugyanazzal az elvvel,
# mint a Whisper/UVR/diarizáció: előtöltjük, hogy első hívásnál ne kelljen
# letölteni.
#
# VÁLTÁS (2026-08): korábban itt az elte-nlp/Racka-4B (magyar-hangolt
# Qwen3-4B) futott, de éles tesztelés során megbízhatatlannak bizonyult —
# visszaidézte a saját rendszer-promptját, duplikálta a válaszokat,
# beleírta a sablon-címkéket a kimenetbe. Ezek ÁLTALÁNOS instrukció-
# követési gyengeségek voltak, nem kifejezetten magyar-nyelvi problémák,
# ezért a felhasználó javaslatára áttértünk a Qwen/Qwen3-4B-Instruct-2507-re:
# ez a Qwen3-4B DEDIKÁLT, CSAK non-thinking módú, 2025 júliusi frissítése —
# architekturálisan képtelen <think> blokkot generálni (nem csak egy
# kikapcsolható flag védi ez ellen), és kifejezetten az instrukció-
# követés, a többnyelvű illeszkedés és a felhasználói preferenciákhoz
# igazodás javítására hangolták. Ugyanaz a 4B méret, ugyanannyi VRAM —
# közvetlen csere. Amit elveszítünk: a Racka kifejezetten magyarra
# hangolt benchmark-előnyét (HuLU/OpenHuEval) — ezt a Qwen3 család
# általános, hivatalosan is erősnek hirdetett többnyelvű instrukció-
# követése/fordítási képessége ellensúlyozza, bár nem magyar-specifikusan
# optimalizált.
#
# Miért snapshot_download, nem model-instantiate (mint a faster-whisper/
# pyannote-nál): a transformers-modelleknél a snapshot_download csak a
# fájlokat tölti le, nem tölti be a modellt memoriaba/GPU-ra — gyorsabb és
# nem terheli feleslegesen a provisioning-folyamatot. Futásidőben a
# tényleges fordítási szolgáltatás majd ugyanebből a cache_dir-ből
# tölt be, újralétöltés nélkül.
#
# Diszkigény: kb. +10-15 GB (NLLB-200-1.3B ~5 GB, Qwen3-4B-Instruct-2507
# ~8 GB fp16-ban) — ha szűkös a hely, ellenőrizd a "df -h /" kimenetet a
# lépés után.
#
# FONTOS: ez a lepes SZANDEKOSAN nem szakitja meg a szkriptet hiba eseten
# (set +e / set -e keretezve). A 7. lepesben a backend mar elindult es
# regisztralva van a supervisornal — ha ez a "nice to have" elotoltes
# halozati vagy diszk-hibaval elhasal, az NE jelentse a teljes
# provisioning bukasat es NE inditson ujra egy teljes, a mar futo
# szolgaltatast megzavaro kort. A hiba csak logolva lesz.
###############################################################################
echo "--- 8. Fordítási modellek előtöltése (NLLB-200 1.3B, Qwen3-4B-Instruct-2507) ---"

set +e

python -m pip install -q sentencepiece accelerate bitsandbytes
if [ $? -ne 0 ]; then
    echo "  FIGYELEM: a forditasi fuggosegek telepitese nem sikerult, a lepes kihagyva." >&2
fi

TRANSLATION_MODEL_DIR="${WORKSPACE_DIR}/models/Translation"
mkdir -p "$TRANSLATION_MODEL_DIR"

python - <<PYEOF
from huggingface_hub import snapshot_download

print("  facebook/nllb-200-1.3B letoltese...")
snapshot_download(
    repo_id="facebook/nllb-200-1.3B",
    cache_dir="${TRANSLATION_MODEL_DIR}",
)

print("  Qwen/Qwen3-4B-Instruct-2507 letoltese...")
snapshot_download(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    cache_dir="${TRANSLATION_MODEL_DIR}",
)
PYEOF
if [ $? -ne 0 ]; then
    echo "  FIGYELEM: a forditasi modellek elotoltese nem sikerult teljesen — a backend ettol meg fut, ez csak a jovobeli forditasi funkciot erinti. Probald ujra kezzel kesobb." >&2
fi

set -e

echo "  Fordítási modellek helye: ${TRANSLATION_MODEL_DIR}"
df -h / | tail -1

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

echo "=== Provisioning kész. A backend a Vast.ai supervisor alatt fut (whisperx-backend, port 8000 — /transcription, /bgm-separation, /vad, /task, /custom-ai/translate, /custom-ai/tts). ==="
