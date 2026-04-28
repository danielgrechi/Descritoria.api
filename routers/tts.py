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
VOZ_TTS = os.environ.get("TTS_VOICE", "eve")
IDIOMA_TTS = os.environ.get("TTS_LANG", "pt")


class TTSRequest(BaseModel):
    texto: str


@router.get("/status", include_in_schema=False)
def status_tts():
    return {
        "motor": "grok-tts",
        "endpoint": "https://api.x.ai/v1/tts",
        "voz": VOZ_TTS,
        "idioma": IDIOMA_TTS,
        "chave_configurada": bool(GROK_TTS_KEY),
        "primeiros_chars": GROK_TTS_KEY[:12] + "..." if GROK_TTS_KEY else "",
    }


@router.post("/falar", summary="Converte texto em fala (Grok TTS)")
async def falar_texto(
    body: TTSRequest,
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    texto = body.texto.strip()
    if not texto:
        raise HTTPException(status_code=422, detail="Texto vazio.")

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://api.x.ai/v1/tts",
                headers={
                    "Authorization": f"Bearer {GROK_TTS_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "text": texto[:4096],
                    "voice_id": VOZ_TTS,
                    "language": IDIOMA_TTS,
                },
            )
            if not resp.is_success:
                log.error("Grok TTS %s: %s", resp.status_code, resp.text)
                raise HTTPException(status_code=502, detail=f"Grok TTS erro {resp.status_code}: {resp.text}")
            return StreamingResponse(
                io.BytesIO(resp.content),
                media_type="audio/mpeg",
                headers={"Content-Disposition": "inline"},
            )
    except httpx.TimeoutException:
        log.warning("Grok TTS timeout")
        raise HTTPException(status_code=504, detail="Grok TTS timeout.")
    except HTTPException:
        raise
    except Exception as e:
        log.error("Grok TTS: %s: %s", type(e).__name__, e)
        raise HTTPException(status_code=502, detail=str(e))
