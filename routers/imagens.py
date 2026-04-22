import base64
import hashlib
import io
import os
from typing import List, Optional

import httpx
import replicate
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session

import models
import schemas
from auth import obter_usuario_atual
from database import get_db

router = APIRouter(prefix="/imagens", tags=["Imagens"])

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")
MODELO_IMAGEM = "yorickvp/llava-13b:80537f9eead1a0bf472503ec4ea45a3799ae5dfd9fac53e39967d1dc6366f8fe"
MODELO_TEXTO = "meta/llama-3.1-8b-instruct"

TIPOS_IMAGEM_PERMITIDOS = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}
FORMATO_PARA_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
}

PROMPT_DESCRICAO = (
    "Você é um assistente de acessibilidade visual para pessoas cegas ou com baixa visão. "
    "Descreva esta imagem em português do Brasil de forma COMPLETA e OBJETIVA. "
    "Inclua obrigatoriamente: "
    "(1) o tema geral e contexto da cena; "
    "(2) todas as pessoas presentes — aparência física, expressões faciais, roupas, posições corporais e interações; "
    "(3) objetos, móveis, animais e elementos do cenário com suas cores, formas, tamanhos e posições relativas; "
    "(4) textos visíveis na imagem; "
    "(5) iluminação, hora do dia e ambiente geral; "
    "(6) qualquer ação ou movimento em curso. "
    "Se houver nudez, conteúdo sexual ou violência, descreva-os com precisão clínica e avise no início. "
    "Seja específico e visual. Escreva em parágrafos fluentes, não em listas."
)

PROMPT_PERGUNTA_TEMPLATE = (
    "Você é um assistente de acessibilidade visual para pessoas cegas. "
    "Contexto da imagem: {contexto}\n\n"
    "Pergunta do usuário: {pergunta}\n\n"
    "Responda em português do Brasil de forma objetiva e detalhada."
)


def validar_imagem(dados: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(dados))
        img.verify()
        img = Image.open(io.BytesIO(dados))
        mime = FORMATO_PARA_MIME.get(img.format)
    except Exception:
        raise HTTPException(status_code=422, detail="Arquivo inválido ou corrompido. Envie uma imagem válida.")
    if not mime or mime not in TIPOS_IMAGEM_PERMITIDOS:
        raise HTTPException(status_code=422, detail=f"Formato não suportado. Use JPEG, PNG, WebP, GIF ou BMP.")
    return mime


def imagem_para_data_uri(dados: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(dados).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def _contexto_pessoas(usuario_id: int, db) -> str:
    """Retorna texto com pessoas conhecidas para incluir no prompt."""
    from models import Pessoa
    pessoas = db.query(Pessoa).filter(Pessoa.usuario_id == usuario_id).all()
    if not pessoas:
        return ""
    lista = "\n".join(f"- {p.nome}: {p.caracteristicas}" for p in pessoas)
    return (
        f"\n\nPESSOAS CONHECIDAS PELO USUÁRIO (use os nomes se reconhecer alguém):\n{lista}\n"
        "Se alguma pessoa na imagem corresponder às características acima, mencione o nome dela na descrição."
    )


def chamar_modelo(data_uri: str, prompt: str) -> str:
    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    saida = client.run(
        MODELO_IMAGEM,
        input={"image": data_uri, "prompt": prompt, "max_new_tokens": 1024, "temperature": 0.2},
    )
    return "".join(saida).strip()


@router.post("/descrever", summary="Descreve uma imagem para usuários com deficiência visual")
async def descrever_imagem(
    arquivo: UploadFile = File(...),
    quantidade: int = Form(1),
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    if quantidade < 1 or quantidade > 3:
        raise HTTPException(status_code=422, detail="O parâmetro 'quantidade' deve ser entre 1 e 3.")

    dados = await arquivo.read()
    if not dados:
        raise HTTPException(status_code=422, detail="Arquivo vazio.")

    mime_type = validar_imagem(dados)
    data_uri = imagem_para_data_uri(dados, mime_type)
    conteudo_hash = hashlib.md5(dados).hexdigest()

    contexto_pessoas = _contexto_pessoas(usuario.id, db)
    prompt_final = PROMPT_DESCRICAO + contexto_pessoas

    try:
        descricoes = [chamar_modelo(data_uri, prompt_final) for _ in range(quantidade)]
    except replicate.exceptions.ReplicateError as e:
        raise HTTPException(status_code=502, detail=f"Erro na API Replicate: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar imagem: {str(e)}")

    # Salva apenas a primeira descrição no histórico
    registro = models.Descricao(
        usuario_id=usuario.id,
        tipo=models.TipoDescricao.foto,
        conteudo_hash=conteudo_hash,
        descricao=descricoes[0],
        modelo=MODELO_IMAGEM,
        formato_original=mime_type,
    )
    db.add(registro)
    db.commit()
    db.refresh(registro)

    return {
        "id": registro.id,
        "descricoes": descricoes,
        "modelo": MODELO_IMAGEM,
        "formato_original": mime_type,
        "tipo": "foto",
        "created_at": registro.created_at,
    }


@router.post("/perguntar", summary="Fazer pergunta sobre uma imagem já descrita")
async def perguntar_sobre_descricao(
    body: schemas.PerguntaRequest,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    registro = db.query(models.Descricao).filter(
        models.Descricao.id == body.descricao_id,
        models.Descricao.usuario_id == usuario.id,
    ).first()
    if not registro:
        raise HTTPException(status_code=404, detail="Descrição não encontrada.")

    prompt = PROMPT_PERGUNTA_TEMPLATE.format(
        contexto=registro.descricao,
        pergunta=body.pergunta,
    )

    try:
        client = replicate.Client(api_token=REPLICATE_API_TOKEN)
        saida = client.run(
            MODELO_TEXTO,
            input={"prompt": prompt, "max_tokens": 512, "temperature": 0.3},
        )
        resposta = "".join(saida).strip()
    except replicate.exceptions.ReplicateError as e:
        raise HTTPException(status_code=502, detail=f"Erro na API Replicate: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

    return {"resposta": resposta, "modelo": MODELO_TEXTO}


@router.post("/perguntar-nova", summary="Anexar nova imagem e fazer pergunta")
async def perguntar_nova_imagem(
    arquivo: UploadFile = File(...),
    pergunta: str = Form(...),
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    dados = await arquivo.read()
    if not dados:
        raise HTTPException(status_code=422, detail="Arquivo vazio.")

    mime_type = validar_imagem(dados)
    data_uri = imagem_para_data_uri(dados, mime_type)

    prompt = (
        f"Você é um assistente de acessibilidade visual para pessoas cegas. "
        f"Analise esta imagem e responda em português do Brasil: {pergunta}"
    )

    try:
        resposta = chamar_modelo(data_uri, prompt)
    except replicate.exceptions.ReplicateError as e:
        raise HTTPException(status_code=502, detail=f"Erro na API Replicate: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

    registro = models.Descricao(
        usuario_id=usuario.id,
        tipo=models.TipoDescricao.foto,
        conteudo_hash=hashlib.md5(dados).hexdigest(),
        descricao=resposta,
        modelo=MODELO_IMAGEM,
        formato_original=mime_type,
    )
    db.add(registro)
    db.commit()
    db.refresh(registro)

    return {"id": registro.id, "resposta": resposta, "modelo": MODELO_IMAGEM}


@router.get("/historico", summary="Histórico de descrições do usuário")
def historico(
    salvo: Optional[bool] = None,
    limite: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    query = db.query(models.Descricao).filter(models.Descricao.usuario_id == usuario.id)
    if salvo is not None:
        query = query.filter(models.Descricao.salvo == salvo)
    total = query.count()
    items = query.order_by(models.Descricao.created_at.desc()).offset(offset).limit(limite).all()
    return {"total": total, "items": [schemas.HistoricoItemResponse.from_orm(i) for i in items]}


@router.post("/salvar/{descricao_id}", summary="Marcar descrição como salva")
def salvar_descricao(
    descricao_id: int,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    registro = db.query(models.Descricao).filter(
        models.Descricao.id == descricao_id,
        models.Descricao.usuario_id == usuario.id,
    ).first()
    if not registro:
        raise HTTPException(status_code=404, detail="Descrição não encontrada.")
    registro.salvo = True
    db.commit()
    return {"mensagem": "Descrição salva com sucesso."}


class DescricaoUrlRequest(BaseModel):
    url: str
    quantidade: int = 1


@router.post("/descrever-url", summary="Descreve imagem a partir de URL da internet")
async def descrever_imagem_url(
    body: DescricaoUrlRequest,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    if body.quantidade < 1 or body.quantidade > 3:
        raise HTTPException(status_code=422, detail="O parâmetro 'quantidade' deve ser entre 1 e 3.")

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resposta_http = await client.get(body.url)
            resposta_http.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=422, detail=f"Erro ao acessar URL: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Não foi possível acessar a URL: {str(e)}")

    dados = resposta_http.content
    content_type = resposta_http.headers.get("content-type", "")

    if "image" not in content_type and not any(
        body.url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"]
    ):
        raise HTTPException(status_code=422, detail="A URL não aponta para uma imagem válida.")

    try:
        mime_type = validar_imagem(dados)
    except HTTPException:
        raise HTTPException(status_code=422, detail="O conteúdo da URL não é uma imagem válida.")

    data_uri = imagem_para_data_uri(dados, mime_type)
    conteudo_hash = hashlib.md5(dados).hexdigest()
    contexto_pessoas = _contexto_pessoas(usuario.id, db)
    prompt_final = PROMPT_DESCRICAO + contexto_pessoas

    try:
        descricoes = [chamar_modelo(data_uri, prompt_final) for _ in range(body.quantidade)]
    except replicate.exceptions.ReplicateError as e:
        raise HTTPException(status_code=502, detail=f"Erro na API Replicate: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

    registro = models.Descricao(
        usuario_id=usuario.id,
        tipo=models.TipoDescricao.foto,
        conteudo_hash=conteudo_hash,
        descricao=descricoes[0],
        modelo=MODELO_IMAGEM,
        formato_original=mime_type,
    )
    db.add(registro)
    db.commit()
    db.refresh(registro)

    return {
        "id": registro.id,
        "descricoes": descricoes,
        "modelo": MODELO_IMAGEM,
        "formato_original": mime_type,
        "tipo": "foto",
        "url_origem": body.url,
        "created_at": registro.created_at,
    }


@router.delete("/historico/{descricao_id}", summary="Remover descrição do histórico")
def deletar_descricao(
    descricao_id: int,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    registro = db.query(models.Descricao).filter(
        models.Descricao.id == descricao_id,
        models.Descricao.usuario_id == usuario.id,
    ).first()
    if not registro:
        raise HTTPException(status_code=404, detail="Descrição não encontrada.")
    db.delete(registro)
    db.commit()
    return {"mensagem": "Descrição removida do histórico."}
