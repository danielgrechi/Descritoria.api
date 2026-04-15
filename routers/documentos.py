import base64
import hashlib
import io
import os

import replicate
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image
from sqlalchemy.orm import Session

import models
from auth import obter_usuario_atual
from database import get_db

router = APIRouter(prefix="/documentos", tags=["Documentos"])

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")
MODELO_DOCUMENTO = "yorickvp/llava-13b"

TIPOS_PERMITIDOS = {
    "image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff",
}

PROMPT_DOCUMENTO = (
    "Você é um leitor de documentos para pessoas cegas. "
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
        client = replicate.Client(api_token=REPLICATE_API_TOKEN)
        saida = client.run(
            MODELO_DOCUMENTO,
            input={"image": data_uri, "prompt": PROMPT_DOCUMENTO, "max_tokens": 2048, "temperature": 0.1},
        )
        transcricao = "".join(saida).strip()
    except replicate.exceptions.ReplicateError as e:
        raise HTTPException(status_code=502, detail=f"Erro na API Replicate: {str(e)}")
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
        "transcricao": transcricao,
        "modelo": MODELO_DOCUMENTO,
        "formato_original": mime_type,
        "tipo": "documento",
        "created_at": registro.created_at,
    }
