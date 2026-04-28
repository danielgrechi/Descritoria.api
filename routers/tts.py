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

GROK_TTS_KEY = os.environ.get("GROK_API_KEY") or os.environ.get("GROK_TTS_KEY", "")
VOZ_TTS = os.environ.get("TTS_VOICE", "Eve")


class TTSRequest(BaseModel):
    texto: str


@router.get("/status", include_in_schema=False)
def status_tts():
    return {
        "motor": "xai-tts",
        "endpoint": "https://api.x.ai/v1/tts",
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

    # Prefixo induz sotaque brasileiro correto na voz Eve
    texto_para_voz = "[fala em português brasileiro, sotaque do Brasil] " + texto[:4000]

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.x.ai/v1/tts",
                headers={
                    "Authorization": f"Bearer {GROK_TTS_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "text": texto_para_voz,
                    "voice_id": VOZ_TTS,
                    "language": "pt-BR",
                    "codec": "mp3",
                    "text_normalization": True,
                },
            )
            if not resp.is_success:
                log.error("xAI TTS %s: %s", resp.status_code, resp.text[:500])
                raise HTTPException(
                    status_code=502,
                    detail=f"xAI TTS erro {resp.status_code}: {resp.text[:200]}",
                )
            return StreamingResponse(
                io.BytesIO(resp.content),
                media_type="audio/mpeg",
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
