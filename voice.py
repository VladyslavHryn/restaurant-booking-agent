"""Voice in and out for the booking agent (the optional * task).

- speak(text)  -> OGG/Opus bytes for a Telegram voice note. Free neural voices via
  edge-tts (no API key): British English by default, Russian/Ukrainian when the
  reply is in that language. Converted with the ffmpeg bundled in imageio-ffmpeg.
- transcribe(audio) -> Heard(text, language, clear, reason), via Whisper on Groq.
  Whisper "hears" words in silence and noise ("Thank you.", "so"), and Groq returns
  no usable no_speech_prob, so the code checks loudness, length, confidence and a
  list of known hallucinations before trusting a transcript.
"""
import asyncio
import os
import re
import tempfile
from dataclasses import dataclass

import edge_tts
import imageio_ffmpeg
from openai import AsyncOpenAI

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
VOICES = {  # first one that works is used
    "en": [os.getenv("TTS_VOICE", "en-GB-RyanNeural"), "en-GB-SoniaNeural"],
    "ru": ["ru-RU-DmitryNeural", "ru-RU-SvetlanaNeural"],
    "uk": ["uk-UA-OstapNeural", "uk-UA-PolinaNeural"],
}

MIN_SECONDS = 0.8        # shorter than this is a mis-tap
SILENCE_DB = -45.0       # mean volume below this is silence
MIN_LOGPROB = -0.6       # Whisper's average log-probability below this is guesswork
HALLUCINATIONS = {       # what Whisper typically "hears" in silence or noise
    "", "you", "so", "thank you", "thanks", "thank you for watching", "thanks for watching",
    "bye", "okay", "oh", "hmm", "um", "uh", ".", "...",
    "субтитры сделал dimatorzok", "продолжение следует", "спасибо за просмотр",
    "редактор субтитров а.семкин корректор а.егорова",
}


@dataclass
class Heard:
    text: str
    language: str      # as Whisper reports it, e.g. "English", "Russian"
    clear: bool        # False: silence, noise, too short or not confident
    reason: str = ""


def language_of(text: str) -> str:
    if re.search("[ієїґІЄЇҐ]", text):
        return "uk"
    if re.search("[а-яА-ЯёЁ]", text):
        return "ru"
    return "en"


async def _run(*args: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await proc.communicate()
    return proc.returncode, out.decode("utf-8", "replace")


async def speak(text: str) -> tuple[bytes, int]:
    """Return (ogg_opus_bytes, duration_seconds), in a voice matching the text's language."""
    with tempfile.TemporaryDirectory() as tmp:
        mp3, ogg = os.path.join(tmp, "a.mp3"), os.path.join(tmp, "a.ogg")
        last_error: Exception | None = None
        for name in VOICES[language_of(text)]:
            try:
                await edge_tts.Communicate(text, name).save(mp3)
                break
            except Exception as e:  # a voice may refuse some text; try the next one
                last_error = e
        else:
            raise RuntimeError(f"text-to-speech failed: {last_error}")
        code, out = await _run(FFMPEG, "-y", "-loglevel", "error", "-i", mp3,
                               "-c:a", "libopus", "-b:a", "32k", "-ac", "1", "-ar", "48000", ogg)
        if code != 0:
            raise RuntimeError(f"ffmpeg failed: {out[:200]}")
        with open(ogg, "rb") as f:
            data = f.read()
    return data, max(1, round(len(data) * 8 / 32000))


async def loudness_db(audio: bytes) -> float:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "in.ogg")
        with open(path, "wb") as f:
            f.write(audio)
        _, out = await _run(FFMPEG, "-hide_banner", "-i", path, "-af", "volumedetect", "-f", "null", "-")
    m = re.search(r"mean_volume:\s*(-?[\d.]+|-inf) dB", out)
    return float(m[1]) if m and m[1] != "-inf" else -99.0


async def transcribe(llm: AsyncOpenAI, audio: bytes) -> Heard:
    # No language and no prompt on purpose: an English prompt (e.g. the restaurant name)
    # made Whisper "hear" Russian speech as English gibberish. Name slips such as
    # "Disham" for "Dishoom" are left to the language model, which reads past them.
    if (db := await loudness_db(audio)) < SILENCE_DB:
        return Heard("", "", False, f"silence ({db:.0f} dB)")
    r = (await llm.audio.transcriptions.create(
        model=STT_MODEL, file=("voice.ogg", audio), response_format="verbose_json")).model_dump()
    text = (r.get("text") or "").strip()
    lang = r.get("language") or ""
    segments = r.get("segments") or []
    logprob = sum(s["avg_logprob"] for s in segments) / len(segments) if segments else -9.0
    normalized = re.sub(r"[^\w\s.]", "", text.lower()).strip(" .")
    if (r.get("duration") or 0) < MIN_SECONDS:
        return Heard(text, lang, False, "too short")
    if normalized in HALLUCINATIONS:
        return Heard(text, lang, False, f"no real speech (Whisper heard {text!r})")
    if logprob < MIN_LOGPROB:
        return Heard(text, lang, False, f"unclear (confidence {logprob:.2f}, heard {text!r})")
    return Heard(text, lang, True)
