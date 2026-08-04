"""
custom_ai/router.py — sajat router: forditas (HTTP-hivas a kulon futo qwen-translate/llama-server-hez, Qwen3.6-27B GGUF, thinking-kepes)
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

F5TTS_HU_DIR = os.environ.get("F5TTS_HU_DIR", "/workspace/models/F5-TTS-Hungarian")

custom_ai_router = APIRouter(prefix="/custom-ai", tags=["Custom AI (Translate + TTS)"])


# ============================================================
# Hang/hatterzene szetvalasztas: BS-Roformer (audio-separator csomag),
# NEM a WhisperX-WebUI beepitett /bgm-separation/-je. Az utobbi csak
# UVR-MDX-NET-Inst_HQ_4/Inst_3 kozott valaszthat (ellenorizve elesben
# a MusicSeparator.available_models-szal) — ezek mar nem SOTA, a
# szetvalasztasnal szovegreszek vesztek el/keveredtek. A BS-Roformer
# (2023, Lu/Wang/Kong/Hung) 2026-ban is a legjobb architektura ehhez —
# az audio-separator (nomadkaraoke) csomag sajat karbantartoja a
# "model_bs_roformer_ep_317_sdr_12.9755.ckpt"-t ajanlja alapertelmezes-
# kent ("go-to a tiszta, teljes spektrumu szetvalasztashoz a legtobb
# bemenethez") — ezt hasznaljuk, nem a csomag sajat (Mel-Band) default-
# jat, mert ez egy nevesitett, indokolt ajanlas.
#
# A modellt az audio_separator maga tolti le automatikusan, elso
# hasznalatkor — nincs kulon snapshot_download lepes.
# ============================================================

@functools.lru_cache
def get_separator():
    from audio_separator.separator import Separator

    # VALOS HIBA ALAPJAN JAVITVA (2026-08): a Separator alapertelmezett
    # model_file_dir-je "/tmp/audio-separator-models/" — ez a rendszer
    # /tmp-je, ami barmikor kiurulhet (instance-ujrainditas, rendszer-
    # takaritas). Ha ez megtortenik, a kovetkezo hivas ujra le kell
    # toltse a ~600MB-os BS-Roformer sulyt a halozatrol, ami megmagyarazza
    # a korabban eszlelt, rejtelyes tobblet-idot (a tenyleges GPU-
    # feldolgozas ~16mp volt, de a teljes lepes ~85mp-ig tartott). Explicit
    # allando helyre (WORKSPACE_DIR-en belul, sosem torlodik automatikusan)
    # iranyitva, hogy ez tobbet ne fordulhasson elo.
    separator_models_dir = os.environ.get(
        "AUDIO_SEPARATOR_MODEL_DIR", "/workspace/models/audio-separator-models"
    )
    os.makedirs(separator_models_dir, exist_ok=True)
    sep = Separator(output_dir=tempfile.gettempdir(), model_file_dir=separator_models_dir)
    sep.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    return sep


@custom_ai_router.post("/separate")
async def separate(audio: UploadFile = File(...)):
    import zipfile

    try:
        separator = get_separator()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"BS-Roformer betoltesi hiba: {exc}")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, audio.filename or "input.wav")
        with open(input_path, "wb") as f:
            f.write(await audio.read())

        try:
            output_files = separator.separate(input_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"BS-Roformer szetvalasztasi hiba: {exc}")

        if not output_files:
            raise HTTPException(status_code=500, detail="A szetvalasztas nem adott vissza kimeneti fajlt.")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for path in output_files:
                # A separator.output_dir-ba irja a kimenetet (nem feltetlenul
                # a tmpdir-be) — abszolut/relativ utat is adhat vissza.
                full_path = path if os.path.isabs(path) else os.path.join(tempfile.gettempdir(), path)
                zf.write(full_path, arcname=os.path.basename(full_path))
        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=separated.zip"},
        )


# ============================================================
# Forditas: Qwen3.6-27B, DE MOST GGUF-kent, llama.cpp/llama-server-en
# keresztul (NEM a teljes bf16 + transformers ut) — a felhasznalo
# kifejezett kerese, mert a bf16 letoltese ~56 GB, a GGUF (Q4_K_M,
# unsloth kiadasa) ~16.8 GB. A GGUF format eretebb/megbizhatobb, mint a
# szort minosegu, nem hivatalos AWQ requant-ok (a Qwen sajat HF-forumjan
# is panaszkodtak ezekre) — unsloth es bartowski a kozossegben
# koztiszteletben allo, sajat KL-divergencia-benchmarkkal osszemert
# GGUF-kiadok.
#
# ARCHITEKTURA-VALTAS: a modell mostantol NEM a backend Python-
# folyamataban, in-process fut, hanem KULON folyamatkent, a
# llama-server-en keresztul (sajat supervisor-program, lasd 8. lepes),
# OpenAI-kompatibilis HTTP API-val. Ez a fuggveny csak egy HTTP-hivast
# csinal, nem tolt be semmit sajat magaban — a "get_stylizer_model()"
# fuggveny es a hozza tartozo transformers/BitsAndBytesConfig import
# EZERT TUNT EL innen.
#
# A JSON_KEZDET/JSON_VEGE kinyeres-logika VALTOZATLAN maradt, es a
# <think>...</think> levagast is megtartottuk VEDEKEZO celbol — bar a
# gondolkodast MOST MAR keresenkent kikapcsoljuk (ld. lent, valos hiba
# alapjan, 2026-08), ha a szerver ezt figyelmen kivul hagyna, a strip
# akkor is arattalanit.
# ============================================================

QWEN_LLAMA_SERVER_URL = os.environ.get("QWEN_LLAMA_SERVER_URL", "http://127.0.0.1:8002")


def _ensure_qwen_translate_running(timeout_seconds: int = 240) -> None:
    """
    ROUTER MOD ota (2026-08) a qwen-translate folyamata MINDIG fut —
    sem a whisperx-backend.sh, sem a provisioning nem allitja le tobbe
    (a korabbi, egyedi stop/start-vezenyles megszunt, ld. 8. lepes
    kommentje). A /health tehat rendszerint AZONNAL valaszol, meg akkor
    is, ha a MODELL maga eppen nincs betoltve (a router-folyamat maga
    konnyu). Ez a fuggveny csak arra az esetre marad meg, ha a folyamat
    valamiert (crash, meg be nem fejezett supervisor-inditas) epp nem
    valaszol — ekkor egyszeruen VAR, ES UJRAPROBALJA, de MAR NEM hiv
    supervisorctl-t (nincs mit "elinditania" — az autostart+autorestart
    ezt magatol kezeli).
    """
    import time

    import requests

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            resp = requests.get(f"{QWEN_LLAMA_SERVER_URL}/health", timeout=3)
            if resp.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(3)

    raise RuntimeError(
        f"A qwen-translate (llama-server) nem valaszol {timeout_seconds} masodpercen belul."
    )


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
    import requests

    _ensure_qwen_translate_running()

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
                "beszélőnként az egész jeleneten át. FONTOS: ez egy MEGLÉVŐ párbeszéd HŰ "
                "fordítása, nem új tartalom írása — a durva kifejezéseket, sértéseket és "
                "trágár szavakat is PONTOSAN, a megfelelő magyar megfelelőjükkel add vissza "
                "(pl. 'idiot' -> 'hülye'/'idióta', ne kerülő, elmosódott vagy kitalált szót "
                "használj helyettük). A regiszter (durvaság szintje) hű megtartása a fordítás "
                "minőségének alapkövetelménye, pont úgy, mint bármely más szónál."
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

    # A kimenet meretenek dinamikus skalazasa a szegmensszammal. VALOS
    # HIBA ALAPJAN VEGLEGESEN ATALAKITVA (2026-08): a korabban bekapcsolt
    # gondolkodas (--reasoning on, szerver-szinten) aranytalanul sok
    # tokent fogyasztott — pl. egy 27 szegmenses, tartalmilag nem
    # bonyolult koteg ~12800 tokent (a teljes max_tokens keretet) elt el
    # PUSZTAN gondolkodasra, ~5 perc alatt, MEGIS ures/hasznalhatatlan
    # valaszt adva vissza. Sikeres, vegigfutott gondolkodo-modu fordítást
    # SOHA nem lattunk ebben a projektben — a felteteizett kontextus/
    # stilus-elony bizonyitatlan maradt, a koltsege (ido, megbizhatatlan-
    # sag) viszont mar most is jol lathato. Ezert MOST, KERESENKENT
    # KIKAPCSOLJUK a gondolkodast a fordításnál ("chat_template_kwargs":
    # {"enable_thinking": False}) — ezt a Qwen3.6 sajat dokumentacioja
    # kifejezetten tamogatja kereskent, a szerver sajat --reasoning on
    # alapbeallitasa MELLETT is felulirhato. Enelkul a max_tokens is
    # sokkal kisebb lehet, mert nincs tobbe gondolkodasi overhead.
    max_tokens = max(2048, 150 + len(segments) * 60)

    try:
        resp = requests.post(
            f"{QWEN_LLAMA_SERVER_URL}/v1/chat/completions",
            json={
                "model": "qwen3.6-27b",
                "messages": messages,
                "max_tokens": max_tokens,
                # A Qwen sajat ajanlasa "Instruct (non-thinking) mode"-hoz:
                # temperature=0.7, top_p=0.80, presence_penalty=1.5 — MAS,
                # mint a thinking-modu ajanlas (1.0/0.95/0.0), mert most
                # kikapcsoljuk a gondolkodast (ld. lent).
                "temperature": 0.7,
                "top_p": 0.80,
                "presence_penalty": 1.5,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=900,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"A qwen-translate (llama-server) hivas sikertelen: {exc}") from exc

    resp_json = resp.json()
    raw_output = resp_json["choices"][0]["message"]["content"].strip()
    finish_reason = resp_json["choices"][0].get("finish_reason", "(ismeretlen)")
    output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

    match = re.search(r"JSON_KEZDET\s*(.+?)\s*JSON_VEGE", output, flags=re.DOTALL)
    if match:
        candidate = match.group(1).strip()
    else:
        # Vedekezo fallback: a jelolok hianyaban probaljuk megkeresni az
        # elso "[" es utolso "]" kozotti reszt.
        start = output.find("[")
        end = output.rfind("]")
        if start == -1 or end == -1 or end <= start:
            # Diagnosztikai celbol beleirjuk a hibauzenetbe a NYERS
            # (think-levagas ELOTTI) valasz elejet/veget, a hosszat, ES a
            # finish_reason-t is ("length" = a max_tokens szakitotta
            # felbe, "stop" = a modell magatol befejezte) — igy legkozelebb
            # NEM kell talalgatni, mi tortent tenylegesen.
            raw_preview = raw_output[:500] if raw_output else "(ures valasz)"
            raise ValueError(
                f"A modell valasza nem tartalmazott ertelmezheto JSON-t. "
                f"max_tokens={max_tokens}, finish_reason={finish_reason!r}, "
                f"nyers valasz hossza={len(raw_output)}, nyers valasz eleje: {raw_preview!r}"
            )
        candidate = output[start : end + 1]

    # Gyakori LLM-JSON hiba: felesleges vesszo a listak/objektumok vegen —
    # ezt levagjuk, mielott parse-olnank.
    candidate = re.sub(r",\s*([\]}])", r"\1", candidate)

    # strict=False: a modell idonkent nyers (nem escape-elt) vezerlo-
    # karaktert (pl. sortorest) tesz egy string-ertek belsejebe — a Python
    # json modulja alapertelmezetten (strict=True) ezt elutasitja
    # ("Invalid control character"), pedig tartalmilag ertelmezheto.
    # strict=False mellett ezeket megengedi.
    parsed = json.loads(candidate, strict=False)
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


@functools.lru_cache
def get_hungarian_asr():
    # Maxdorger29/whisper-large-v3-hungarian-lora — a F5-TTS-Hungarian
    # SZERZOJE altal, KIFEJEZETTEN erre a celra epitett ASR-modell:
    # "milliméter-pontos" ref_text eloallitasa a F5-TTS-hez, mert az
    # "erosen erzekeny az atirat-pontatlansagokra" (sajat modellkartya).
    # User kerese (2026-08): ha a /tts hivas nem kap explicit ref_text-et,
    # ez transzkribalja a ref_audio-t a TENYLEGES hangbol — ez pontosabb
    # lehet, mint egy kezzel begepelt (akar szo szerinti) atirat, mert a
    # tenyleges kiejtes/szunetek/hangsulyok alapjan keszul, nem irott
    # szovegbol.
    from faster_whisper import WhisperModel

    return WhisperModel("Maxdorger29/whisper-large-v3-hungarian-lora", device="cuda", compute_type="float16")


@custom_ai_router.post("/tts")
async def tts(
    ref_audio: UploadFile = File(..., description="Referencia hangminta (5-15 mp, pl. speaker_00.wav)"),
    ref_text: str = Form(
        "",
        description=(
            "A referencia hang pontos átirata. Ha üresen hagyod ÉS az "
            "auto_transcribe_ref=true, a rendszer automatikusan transzkribálja "
            "a ref_audio-t a Maxdorger29/whisper-large-v3-hungarian-lora "
            "modellel — de ez EGY TOVÁBBI modellt tölt be VRAM-ba a Whisper és "
            "F5-TTS MELLÉ, ezért nem automatikus, kifejezetten be kell kapcsolni."
        ),
    ),
    gen_text: str = Form(..., description="A felolvasandó/generálandó magyar szöveg"),
    auto_transcribe_ref: bool = Form(
        False,
        description=(
            "Ha true ÉS ref_text üres, automatikusan transzkribálja a "
            "ref_audio-t a magyar ASR LoRA-val. VALÓS HIBA ALAPJAN (2026-08): "
            "korábban ez AUTOMATIKUSAN lefutott üres ref_text eseten, es a "
            "fo pipeline-ban VRAM-OOM-ot okozott (harmadik modell betoltese "
            "a mar bent levo Whisper+F5-TTS melle, ugyanabban a folyamatban). "
            "Mostantol kifejezett bekapcsolas kell — a run_pipeline.php SOHA "
            "nem kuldi ezt true-ra, csak kezi/curl tesztekhez valo."
        ),
    ),
    # ========================================================
    # ALÁBBI PARAMÉTEREK (2026-08): a F5TTS.infer() TÉNYLEGES,
    # ellenőrzött szignatúrája (SWivid/F5-TTS api.py, hitelesített
    # commit-diffből, NEM feltételezés) — mindegyik PONTOSAN a
    # könyvtár saját alapértékén all, ha nem küldöd őket, tehát a
    # viselkedés a dokumentált alapbeállítással EGYEZIK. User kifejezett
    # kérése: ne "varázsoljunk" a szerveroldalon, hanem a TELJES
    # paraméterfelület legyen közvetlenül tesztelhető curl-lel, gyors
    # iterációhoz, a lassú, teljes pipeline megkerülésével.
    # ========================================================
    target_rms: float = Form(0.1, description="Célzott RMS hangerő-normalizáláshoz (könyvtár alapértéke: 0.1)."),
    cross_fade_duration: float = Form(0.15, description="Átfedés (mp) a batch-ek/szegmensek között (alapérték: 0.15)."),
    sway_sampling_coef: float = Form(-1.0, description="Sway sampling együttható a flow-matching mintavételezéshez (alapérték: -1)."),
    cfg_strength: float = Form(2.0, description="Classifier-free guidance erőssége — magasabb: stabilabb, kevésbé kifejező (alapérték: 2)."),
    nfe_step: int = Form(32, description="ODE-lépések száma — magasabb: jobb minőség, lassabb (alapérték: 32; 16=gyors, 64=legjobb minőség)."),
    speed: float = Form(1.0, description="Sebesség-szorzó (alapérték: 1.0 — a könyvtár saját alapértéke)."),
    fix_duration: Optional[float] = Form(None, description="Explicit célhossz mp-ben (referencia+generált együtt). Alapból None (auto-becslés)."),
    remove_silence: bool = Form(False, description="A generált hang végi/eleji csend eltávolítása utófeldolgozással (alapérték: False)."),
    use_official_preprocessing: bool = Form(
        True,
        description=(
            "VALÓS, FORRÁSBÓL IGAZOLT ELTÉRÉS (2026-08): a Gradio-app/CLI "
            "MINDIG meghívja a f5_tts.infer.utils_infer.preprocess_ref_audio_text() "
            "függvényt a generálás előtt — ez levágja a csendet a referencia "
            "hang elejéről/végéről, pontosan 50ms csendet ad a végéhez (NEM "
            "1mp-et, mint a mi korábbi SAMPLE_PAD_SECONDS-unk), 12mp fölött "
            "vágja a klipet, és biztosítja hogy a ref_text mondatzáró "
            "írásjelre/szóközre végződjön. Ezt korábban SOHA nem hívtuk. "
            "Alapból BE van kapcsolva (true) — állítsd false-ra, ha ki "
            "akarod hagyni, összehasonlításhoz."
        ),
    ),
):
    import soundfile as sf

    model = get_f5tts()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_ref:
        tmp_ref.write(await ref_audio.read())
        tmp_ref_path = tmp_ref.name

    try:
        effective_ref_text = ref_text.strip()
        if not effective_ref_text:
            if not auto_transcribe_ref:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Nincs ref_text megadva, es auto_transcribe_ref sincs "
                        "bekapcsolva. Vagy adj meg explicit ref_text-et, vagy "
                        "kuldd auto_transcribe_ref=true-t (VRAM-koltseges — "
                        "kulon ASR-modellt tolt be)."
                    ),
                )
            try:
                asr = get_hungarian_asr()
                segments, _info = asr.transcribe(tmp_ref_path, language="hu")
                effective_ref_text = " ".join(seg.text.strip() for seg in segments).strip()
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Automatikus ref_text-atirat (Hungarian ASR LoRA) sikertelen: {exc}",
                )
            if not effective_ref_text:
                raise HTTPException(
                    status_code=400,
                    detail="Az automatikus atirat ures lett — adj meg explicit ref_text-et.",
                )

        effective_ref_path = tmp_ref_path
        if use_official_preprocessing:
            from f5_tts.infer.utils_infer import preprocess_ref_audio_text

            effective_ref_path, effective_ref_text = preprocess_ref_audio_text(
                tmp_ref_path, effective_ref_text, show_info=lambda *a, **kw: None
            )

        wav, sr, _ = model.infer(
            ref_file=effective_ref_path,
            ref_text=effective_ref_text,
            gen_text=gen_text,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            sway_sampling_coef=sway_sampling_coef,
            cfg_strength=cfg_strength,
            nfe_step=nfe_step,
            speed=speed,
            fix_duration=fix_duration,
            remove_silence=remove_silence,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"F5-TTS generálási hiba: {exc}")
    finally:
        os.unlink(tmp_ref_path)

    buffer = io.BytesIO()
    sf.write(buffer, wav, sr, format="WAV")
    buffer.seek(0)
    from urllib.parse import quote

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"X-Effective-Ref-Text": quote(effective_ref_text)},
    )


# ============================================================
# Magyar szoveg-normalizalas F5-TTS-hez — a sarpba/hun repo SAJAT,
# klonozott forraskodjabol (NEM sajat irasu — user kifejezett kerese:
# csak tenylegesen letezo, ellenorzott kodot hasznaljunk, ne sajat
# talalmanyt). A normalize() fuggveny sorszamokat/datumokat/idopontokat
# ir at szoveggé, mozaikszavakat betuz, es KISBETUSSE alakitja + "... "
# elotagot ad a szoveghez — ez a repo SAJAT, dokumentalt, ELOIRT
# mukodese (nem mellekhatas), a modell sajat mintai is mind kisbetusek.
# Forras: https://github.com/sarpba/hun
# ============================================================

@functools.lru_cache
def get_hun_normalizer():
    import sys

    normalizer_dir = os.environ.get("HUN_NORMALIZER_DIR", "/workspace/models/hun-normalizer")
    if normalizer_dir not in sys.path:
        sys.path.insert(0, normalizer_dir)
    from normaliser import normalize

    return normalize


class NormalizeTextRequest(BaseModel):
    text: str


class NormalizeTextResponse(BaseModel):
    normalized_text: str


@custom_ai_router.post("/normalize_text", response_model=NormalizeTextResponse)
def normalize_text(req: NormalizeTextRequest):
    try:
        normalize_fn = get_hun_normalizer()
        result = normalize_fn(req.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Szoveg-normalizalasi hiba: {exc}")
    return NormalizeTextResponse(normalized_text=result)


@custom_ai_router.get("/health")
def health():
    return {"status": "ok"}
