"""
Stage 1: Speech-to-Text.

Only invoked when the incoming request is voice (blueprint 2.1). Uses
Sarvam's saaras:v3 model, which transcribes Hindi/regional-language speech
directly (blueprint says the system "can understand spoken Hindi/regional
languages and convert it to text" - Part 1).

Kept as a thin, swappable client: nothing downstream depends on Sarvam
specifically, only on this function returning plain text.
"""
from __future__ import annotations

import base64
import io
import logging

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)


class STTError(Exception):
    pass


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def transcribe_audio(audio_base64: str, language_hint: str | None = None) -> str:
    """
    Send base64-encoded audio to Sarvam's speech-to-text endpoint and return
    the transcript. Raises STTError on failure after retries.
    """
    if not settings.sarvam_api_key:
        raise STTError(
            "SARVAM_API_KEY is not configured. Voice input is out of scope "
            "until this is set (blueprint 2.6: needed only if voice input is in scope)."
        )

    audio_bytes = base64.b64decode(audio_base64)
    files = {"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")}
    data = {"model": settings.sarvam_stt_model}
    if language_hint:
        data["language_code"] = language_hint

    headers = {"api-subscription-key": settings.sarvam_api_key}

    resp = requests.post(
        f"{settings.sarvam_base_url}/speech-to-text",
        headers=headers,
        files=files,
        data=data,
        timeout=30,
    )
    if resp.status_code != 200:
        logger.error("Sarvam STT failed: %s %s", resp.status_code, resp.text)
        raise STTError(f"Sarvam STT request failed with status {resp.status_code}")

    payload = resp.json()
    transcript = payload.get("transcript", "")
    if not transcript:
        raise STTError("Sarvam STT returned an empty transcript")

    return transcript.strip()
