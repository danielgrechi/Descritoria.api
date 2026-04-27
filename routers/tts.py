import io
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI, OpenAIError
from pydantic import BaseModel

import models
from auth import obter_usuario_atual

router = APIRouter(prefix="/tts", tags=["TTS"])

GROK_TTS_KEY = os.environ.get("GROK_TTS_KEY", os.environ.get("GROK_API_KEY", ""))
MODELO_TTS = "grok-tts"
VOZ_TTS = os.environ.get("TTS_VOICE", "Aria")


class TTSRequest(BaseModel):
    texto: str


def _tts_client() -> OpenAI:
    return OpenAI(api_key=GROK_TTS_KEY, base_url="https://api.x.ai/v1")


@router.post("/falar", summary="Converte texto em fala (Grok TTS)")
async def falar_texto(
    body: TTSRequest,
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    texto = body.texto.strip()
    if not texto:
        raise HTTPException(status_code=422, detail="Texto vazio.")
    try:
        client = _tts_client()
        response = client.audio.speech.create(
            model=MODELO_TTS,
            voice=VOZ_TTS,
            input=texto[:4096],
        )
        audio_bytes = response.read()
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Erro no TTS Grok: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno TTS: {str(e)}")

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline"},
    )
