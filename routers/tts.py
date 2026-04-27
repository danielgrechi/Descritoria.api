import asyncio
import base64
import io
import json
import os
import struct

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import models
from auth import obter_usuario_atual

router = APIRouter(prefix="/tts", tags=["TTS"])

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


async def _tts_via_realtime(texto: str) -> bytes:
    import websockets

    auth_header = [("Authorization", f"Bearer {GROK_TTS_KEY}")]
    chunks: list[bytes] = []

    # Tenta new API (websockets 12+) com additional_headers
    try:
        ctx = websockets.connect(REALTIME_URL, additional_headers=auth_header)
    except TypeError:
        # Fallback para legacy API com extra_headers
        ctx = websockets.connect(REALTIME_URL, extra_headers=auth_header)

    async with ctx as ws:
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "voice": VOZ_TTS,
                "modalities": ["audio", "text"],
                "instructions": "Leia o texto exatamente como fornecido.",
            },
        }))
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": texto}],
            },
        }))
        await ws.send(json.dumps({"type": "response.create"}))

        async for raw in ws:
            if isinstance(raw, bytes):
                chunks.append(raw)
                continue
            event = json.loads(raw)
            etype = event.get("type", "")
            print(f"[TTS] {etype}", flush=True)
            if etype == "response.audio.delta":
                delta = event.get("delta", "")
                if delta:
                    chunks.append(base64.b64decode(delta))
            elif etype in ("response.done", "error"):
                if etype == "error":
                    print(f"[TTS] ERRO API: {event}", flush=True)
                break

    raw_audio = b"".join(chunks)
    if not raw_audio:
        raise RuntimeError("API não retornou áudio")
    return _pcm16_para_wav(raw_audio)


@router.post("/falar", summary="Converte texto em fala (xAI Realtime Voice)")
async def falar_texto(
    body: TTSRequest,
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    texto = body.texto.strip()
    if not texto:
        raise HTTPException(status_code=422, detail="Texto vazio.")
    try:
        audio = await asyncio.wait_for(_tts_via_realtime(texto[:1000]), timeout=12.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Timeout no TTS.")
    except Exception as e:
        print(f"[TTS] ERRO FINAL: {type(e).__name__}: {e}", flush=True)
        raise HTTPException(status_code=502, detail=f"Erro no TTS: {str(e)}")

    return StreamingResponse(
        io.BytesIO(audio),
        media_type="audio/wav",
        headers={"Content-Disposition": "inline"},
    )
