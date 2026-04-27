import asyncio
import io
import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import models
from auth import obter_usuario_atual

router = APIRouter(prefix="/tts", tags=["TTS"])
log = logging.getLogger("descritoria.tts")

VOZ_TTS = os.environ.get("TTS_VOICE", "pt-BR-FranciscaNeural")


class TTSRequest(BaseModel):
    texto: str


async def _tts_edge(texto: str) -> bytes:
    import edge_tts
    communicate = edge_tts.Communicate(texto, voice=VOZ_TTS)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    data = buf.getvalue()
    if not data:
        raise RuntimeError("edge-tts não retornou áudio")
    return data


@router.get("/status", summary="Status do TTS", include_in_schema=False)
def status_tts():
    return {"motor": "edge-tts", "voz": VOZ_TTS, "ok": True}


@router.post("/falar", summary="Converte texto em fala")
async def falar_texto(
    body: TTSRequest,
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    texto = body.texto.strip()
    if not texto:
        raise HTTPException(status_code=422, detail="Texto vazio.")
    try:
        audio = await asyncio.wait_for(_tts_edge(texto[:1000]), timeout=15.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Timeout no TTS.")
    except Exception as e:
        log.error("TTS falhou: %s: %s", type(e).__name__, e)
        raise HTTPException(status_code=502, detail=f"Erro no TTS: {str(e)}")

    return StreamingResponse(
        io.BytesIO(audio),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline"},
    )
