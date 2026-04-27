import asyncio
import base64
import io
import json
import logging
import os
import struct

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI, OpenAIError
from pydantic import BaseModel

import models
from auth import obter_usuario_atual

router = APIRouter(prefix="/tts", tags=["TTS"])
log = logging.getLogger("descritoria.tts")

GROK_TTS_KEY = os.environ.get("GROK_TTS_KEY", os.environ.get("GROK_API_KEY", ""))
VOZ_TTS = os.environ.get("TTS_VOICE", "Eve")
REALTIME_URL = "wss://api.x.ai/v1/realtime"


class TTSRequest(BaseModel):
    texto: str


def _pcm16_para_wav(pcm: bytes, rate: int = 24000) -> bytes:
    size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + size, b"WAVE",
        b"fmt ", 16, 1, 1, rate,
        rate * 2, 2, 16,
        b"data", size,
    )
    return header + pcm


async def _tts_rest(texto: str) -> bytes:
    """Tenta endpoint REST /v1/audio/speech (compatível com OpenAI)."""
    client = OpenAI(api_key=GROK_TTS_KEY, base_url="https://api.x.ai/v1", timeout=20.0)
    response = client.audio.speech.create(
        model="grok-tts",
        voice=VOZ_TTS,
        input=texto,
    )
    return response.read()


async def _tts_websocket(texto: str) -> bytes:
    """Tenta endpoint Realtime via WebSocket."""
    import websockets

    auth = [("Authorization", f"Bearer {GROK_TTS_KEY}")]
    chunks: list[bytes] = []

    connect_kwargs = {"additional_headers": auth}
    try:
        ctx = websockets.connect(REALTIME_URL, **connect_kwargs)
        ctx.__aenter__  # testa se é válido
    except (TypeError, AttributeError):
        connect_kwargs = {"extra_headers": auth}
        ctx = websockets.connect(REALTIME_URL, **connect_kwargs)

    async with ctx as ws:
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {"voice": VOZ_TTS, "modalities": ["audio", "text"],
                        "instructions": "Leia o texto exatamente como fornecido."},
        }))
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": texto}]},
        }))
        await ws.send(json.dumps({"type": "response.create"}))

        async for raw in ws:
            if isinstance(raw, bytes):
                chunks.append(raw)
                continue
            event = json.loads(raw)
            etype = event.get("type", "")
            log.info("evento WS: %s", etype)
            if etype == "response.audio.delta":
                delta = event.get("delta", "")
                if delta:
                    chunks.append(base64.b64decode(delta))
            elif etype in ("response.done", "error"):
                if etype == "error":
                    log.error("API error: %s", event)
                break

    raw_audio = b"".join(chunks)
    if not raw_audio:
        raise RuntimeError("API não retornou áudio")
    return _pcm16_para_wav(raw_audio)


@router.get("/status", summary="Status do TTS", include_in_schema=False)
def status_tts():
    return {
        "chave_configurada": bool(GROK_TTS_KEY),
        "primeiros_chars": GROK_TTS_KEY[:8] + "..." if GROK_TTS_KEY else "",
        "voz": VOZ_TTS,
    }


@router.post("/falar", summary="Converte texto em fala (xAI TTS)")
async def falar_texto(
    body: TTSRequest,
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    texto = body.texto.strip()
    if not texto:
        raise HTTPException(status_code=422, detail="Texto vazio.")

    # Tenta REST primeiro (mais simples), depois WebSocket
    for tentativa, fn in [("REST", _tts_rest), ("WebSocket", _tts_websocket)]:
        try:
            audio = await asyncio.wait_for(fn(texto[:1000]), timeout=20.0)
            log.info("TTS via %s: %d bytes", tentativa, len(audio))
            media = "audio/mpeg" if tentativa == "REST" else "audio/wav"
            return StreamingResponse(
                io.BytesIO(audio),
                media_type=media,
                headers={"Content-Disposition": "inline"},
            )
        except asyncio.TimeoutError:
            log.warning("TTS %s: timeout", tentativa)
        except Exception as e:
            log.error("TTS %s falhou: %s: %s", tentativa, type(e).__name__, e)

    raise HTTPException(status_code=502, detail="Todas as tentativas de TTS falharam.")
