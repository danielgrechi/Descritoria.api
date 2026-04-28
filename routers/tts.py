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

# Prefere a chave principal (que funciona para imagens) sobre a chave de voz agente
GROK_TTS_KEY = os.environ.get("GROK_API_KEY") or os.environ.get("GROK_TTS_KEY", "")
MODELO_TTS = os.environ.get("TTS_MODEL", "grok-tts-preview")
VOZ_TTS = os.environ.get("TTS_VOICE", "Eve")


class TTSRequest(BaseModel):
    texto: str


@router.get("/status", include_in_schema=False)
def status_tts():
    return {
        "motor": "xai-tts",
        "endpoint": "https://api.x.ai/v1/audio/speech",
        "modelo": MODELO_TTS,
        "voz": VOZ_TTS,
        "chave_configurada": bool(GROK_TTS_KEY),
        "primeiros_chars": GROK_TTS_KEY[:12] + "..." if GROK_TTS_KEY else "",
    }


@router.post("/falar", summary="Converte texto em fala (xAI TTS)")
async def falar_texto(
    body: TTSRequest,
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    texto = body.texto.strip()
    if not texto:
        raise HTTPException(status_code=422, detail="Texto vazio.")

    if not GROK_TTS_KEY:
        raise HTTPException(status_code=503, detail="Chave de API não configurada.")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.x.ai/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {GROK_TTS_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODELO_TTS,
                    "input": texto[:4096],
                    "voice": VOZ_TTS,
                },
            )
            if not resp.is_success:
                log.error("xAI TTS %s: %s", resp.status_code, resp.text[:500])
                raise HTTPException(
                    status_code=502,
                    detail=f"xAI TTS erro {resp.status_code}: {resp.text[:200]}",
                )
            content_type = resp.headers.get("content-type", "audio/mpeg")
            return StreamingResponse(
                io.BytesIO(resp.content),
                media_type=content_type,
                headers={"Content-Disposition": "inline"},
            )
    except httpx.TimeoutException:
        log.warning("xAI TTS timeout")
        raise HTTPException(status_code=504, detail="xAI TTS timeout.")
    except HTTPException:
        raise
    except Exception as e:
        log.error("xAI TTS: %s: %s", type(e).__name__, e)
        raise HTTPException(status_code=502, detail=str(e))
