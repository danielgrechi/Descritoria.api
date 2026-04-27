import base64
import hashlib
import io
import os

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from openai import OpenAI, OpenAIError
from PIL import Image
from sqlalchemy.orm import Session

import models
from auth import obter_usuario_atual
from database import get_db

router = APIRouter(prefix="/documentos", tags=["Documentos"])

GROK_API_KEY = os.environ.get("GROK_API_KEY", "")
MODELO_DOCUMENTO = "grok-2-vision-latest"

TIPOS_PERMITIDOS = {
    "image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff",
}

PROMPT_DOCUMENTO = (
    "Você é um leitor de documentos para pessoas cegas no Brasil. "
    "Leia e transcreva em português do Brasil TODO o texto visível nesta imagem, "
    "preservando a estrutura (parágrafos, listas, títulos). "
    "Após a transcrição, descreva brevemente o layout do documento "
    "(ex: formulário, carta, recibo, placa, etc). "
    "Se não houver texto, descreva o conteúdo visual da imagem normalmente."
)

FORMATO_PARA_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
}


def _grok_client() -> OpenAI:
    return OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")


def validar_arquivo(dados: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(dados))
        img.verify()
        img = Image.open(io.BytesIO(dados))
        mime = FORMATO_PARA_MIME.get(img.format)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Arquivo inválido. Envie uma imagem do documento (JPEG, PNG, WebP, BMP ou TIFF).",
        )
    if not mime or mime not in TIPOS_PERMITIDOS:
        raise HTTPException(
            status_code=422,
            detail=f"Formato não suportado. Use JPEG, PNG, WebP, BMP ou TIFF.",
        )
    return mime


@router.post("/ler", summary="Lê e transcreve o texto de um documento ou imagem")
async def ler_documento(
    arquivo: UploadFile = File(...),
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    dados = await arquivo.read()
    if not dados:
        raise HTTPException(status_code=422, detail="Arquivo vazio.")

    mime_type = validar_arquivo(dados)
    b64 = base64.b64encode(dados).decode("utf-8")
    data_uri = f"data:{mime_type};base64,{b64}"
    conteudo_hash = hashlib.md5(dados).hexdigest()

    try:
        client = _grok_client()
        response = client.chat.completions.create(
            model=MODELO_DOCUMENTO,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": PROMPT_DOCUMENTO},
                ],
            }],
            max_tokens=2048,
            temperature=0.1,
        )
        transcricao = response.choices[0].message.content.strip()
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Erro na API Grok: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

    registro = models.Descricao(
        usuario_id=usuario.id,
        tipo=models.TipoDescricao.documento,
        conteudo_hash=conteudo_hash,
        descricao=transcricao,
        modelo=MODELO_DOCUMENTO,
        formato_original=mime_type,
    )
    db.add(registro)
    db.commit()
    db.refresh(registro)

    return {
        "id": registro.id,
        "texto": transcricao,
        "modelo": MODELO_DOCUMENTO,
        "formato_original": mime_type,
        "tipo": "documento",
        "created_at": registro.created_at,
    }
