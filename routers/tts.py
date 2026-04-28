import asyncio
import io
import logging
import os

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import models
from auth import obter_usuario_atual

router = APIRouter(prefix="/tts", tags=["TTS"])
log = logging.getLogger("descritoria.tts")

GROK_TTS_KEY = os.environ.get("GROK_TTS_KEY", os.environ.get("GROK_API_KEY", ""))
VOZ_TTS = os.environ.get("TTS_VOICE", "eve")   # eve, ara, rex, sal, leo
IDIOMA_TTS = os.environ.get("TTS_LANG", "pt")


class TTSRequest(BaseModel):
    texto: str


async def _tts_grok(texto: str) -> bytes:
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            "https://api.x.ai/v1/tts",
            headers={
                "Authorization": f"Bearer {GROK_TTS_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "text": texto,
                "voice_id": VOZ_TTS,
                "language": IDIOMA_TTS,
            },
        )
        resp.raise_for_status()
        return resp.content


async def _tts_edge_fallback(texto: str) -> bytes:
    import edge_tts
    communicate = edge_tts.Communicate(texto, voice="pt-BR-FranciscaNeural")
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    data = buf.getvalue()
    if not data:
        raise RuntimeError("edge-tts sem áudio")
    return data


@router.get("/status", summary="Status do TTS", include_in_schema=False)
def status_tts():
    return {
        "motor_principal": "grok-tts (api.x.ai/v1/tts)",
        "voz": VOZ_TTS,
        "idioma": IDIOMA_TTS,
        "chave_configurada": bool(GROK_TTS_KEY),
        "primeiros_chars": GROK_TTS_KEY[:8] + "..." if GROK_TTS_KEY else "",
        "fallback": "edge-tts pt-BR-FranciscaNeural",
    }


@router.post("/falar", summary="Converte texto em fala (Grok TTS)")
async def falar_texto(
    body: TTSRequest,
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    texto = body.texto.strip()
    if not texto:
        raise HTTPException(status_code=422, detail="Texto vazio.")

    # Tenta Grok TTS primeiro
    try:
        audio = await asyncio.wait_for(_tts_grok(texto[:4096]), timeout=20.0)
        log.info("Grok TTS: %d bytes", len(audio))
        return StreamingResponse(
            io.BytesIO(audio),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline"},
        )
    except httpx.HTTPStatusError as e:
        log.error("Grok TTS HTTP %s: %s", e.response.status_code, e.response.text)
    except asyncio.TimeoutError:
        log.warning("Grok TTS timeout")
    except Exception as e:
        log.error("Grok TTS erro: %s: %s", type(e).__name__, e)

    # Fallback: edge-tts
    try:
        audio = await asyncio.wait_for(_tts_edge_fallback(texto[:4096]), timeout=15.0)
        log.info("edge-tts fallback: %d bytes", len(audio))
        return StreamingResponse(
            io.BytesIO(audio),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline"},
        )
    except Exception as e:
        log.error("edge-tts falhou: %s", e)
        raise HTTPException(status_code=502, detail="TTS indisponível.")
