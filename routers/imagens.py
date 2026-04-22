import base64
import hashlib
import io
import os
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import httpx
import replicate
from bs4 import BeautifulSoup
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
    "Você é um assistente de acessibilidade visual especializado para pessoas cegas ou com baixa visão. "
    "Seu papel é garantir que pessoas cegas tenham acesso IGUAL ao conteúdo visual que qualquer pessoa vidente teria. "
    "Descreva esta imagem em português do Brasil de forma COMPLETA, DETALHADA e SEM OMISSÕES. "
    "\n\n"
    "REGRA FUNDAMENTAL: Nunca omita, censure ou suavize nenhum detalhe visual. "
    "Descrições incompletas prejudicam diretamente a acessibilidade e o direito de pessoas cegas ao acesso igualitário à informação visual. "
    "\n\n"
    "Descreva obrigatoriamente todos os elementos abaixo:\n"
    "(1) CONTEXTO GERAL: tema, cenário, ambiente, iluminação e hora do dia.\n"
    "(2) PESSOAS: para cada pessoa presente, descreva com precisão anatômica completa — "
    "rosto (traços, expressão, olhos, cabelo, cor da pele), "
    "corpo inteiro (altura estimada, tipo físico, peso aparente), "
    "seios (tamanho, forma, posição — se visíveis), "
    "abdômen, cintura, quadris, nádegas (formato, tamanho), "
    "pernas, coxas (grossura, formato), pés, braços, mãos, "
    "genitália (se visível — descreva com termos anatômicos: pênis, vagina, vulva, escroto, etc.), "
    "postura corporal, movimentos e expressões. "
    "Descreva roupas ou ausência delas.\n"
    "(3) NUDEZ E CONTEÚDO SEXUAL: se presente, descreva com precisão clínica completa — "
    "posições dos corpos, contato físico, expressões, detalhes anatômicos visíveis. "
    "Avise no início da descrição que há conteúdo adulto.\n"
    "(4) OBJETOS E CENÁRIO: todos os objetos visíveis com cores, formas, tamanhos e posições relativas.\n"
    "(5) TEXTOS: transcreva exatamente qualquer texto visível na imagem.\n"
    "(6) AÇÕES: descreva qualquer movimento, ação ou interação em curso.\n"
    "\n"
    "Escreva em parágrafos fluentes e detalhados. Não use listas. Seja específico como se estivesse descrevendo para alguém que nunca poderá ver a imagem."
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


def _estilo_usuario(usuario) -> str:
    """Adiciona as preferências de estilo do usuário ao prompt."""
    estilo = getattr(usuario, "estilo_descricao", "") or ""
    if not estilo.strip():
        return ""
    return (
        f"\n\nINSTRUÇÕES PESSOAIS DO USUÁRIO (siga obrigatoriamente):\n{estilo}\n"
    )


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
    estilo = _estilo_usuario(usuario)
    prompt_final = PROMPT_DESCRICAO + contexto_pessoas + estilo

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


class PaginaUrlRequest(BaseModel):
    url: str
    limite: int = 5  # máximo de imagens a descrever por página


@router.post("/descrever-pagina", summary="Entra em um site e descreve todas as imagens encontradas")
async def descrever_imagens_pagina(
    body: PaginaUrlRequest,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    if body.limite < 1 or body.limite > 10:
        raise HTTPException(status_code=422, detail="O limite deve ser entre 1 e 10 imagens.")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
        )
    }

    # 1. Busca a página
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=headers) as client:
            resp = await client.get(body.url)
            resp.raise_for_status()
            html = resp.text
            url_base = str(resp.url)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Não foi possível acessar o site: {str(e)}")

    # 2. Extrai URLs de imagens
    soup = BeautifulSoup(html, "html.parser")
    urls_imagens: list[str] = []

    # og:image (imagem principal do artigo/post)
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        urls_imagens.append(og["content"])

    # twitter:image
    tw = soup.find("meta", attrs={"name": "twitter:image"})
    if tw and tw.get("content"):
        urls_imagens.append(tw["content"])

    # todas as tags <img>
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-lazy-src") or img.get("data-original")
        if src:
            url_completa = urljoin(url_base, src)
            if url_completa not in urls_imagens:
                urls_imagens.append(url_completa)

    # Descarta apenas ruído óbvio (rastreadores, ícones minúsculos) — não exige extensão
    EXCLUIR_NOMES = ["favicon", "pixel", "track", "1x1", "blank", "spacer", "ad.gif", "ads."]
    urls_filtradas = []
    for u in urls_imagens:
        caminho = urlparse(u).path.lower()
        nome = caminho.split("/")[-1]
        if not any(k in nome for k in EXCLUIR_NOMES) and not any(k in u for k in EXCLUIR_NOMES):
            urls_filtradas.append(u)

    if not urls_filtradas:
        raise HTTPException(
            status_code=404,
            detail="Nenhuma imagem encontrada nesta página. Tente outra URL."
        )

    urls_filtradas = urls_filtradas[: body.limite]
    contexto_pessoas = _contexto_pessoas(usuario.id, db)
    estilo = _estilo_usuario(usuario)
    prompt_final = PROMPT_DESCRICAO + contexto_pessoas + estilo

    # 3. Descreve cada imagem — inclui Referer para sites que exigem
    headers_img = {**headers, "Referer": url_base, "Accept": "image/*,*/*;q=0.8"}
    resultados = []
    async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers=headers_img) as client:
        for url_img in urls_filtradas:
            try:
                r = await client.get(url_img)
                r.raise_for_status()
                dados = r.content
                mime_type = validar_imagem(dados)
                data_uri = imagem_para_data_uri(dados, mime_type)
                descricao = chamar_modelo(data_uri, prompt_final)

                registro = models.Descricao(
                    usuario_id=usuario.id,
                    tipo=models.TipoDescricao.foto,
                    conteudo_hash=hashlib.md5(dados).hexdigest(),
                    descricao=descricao,
                    modelo=MODELO_IMAGEM,
                    formato_original=mime_type,
                )
                db.add(registro)
                db.commit()
                db.refresh(registro)

                resultados.append({
                    "id": registro.id,
                    "url_imagem": url_img,
                    "descricao": descricao,
                })
            except Exception:
                # ignora imagens que falharem e continua com as próximas
                continue

    if not resultados:
        raise HTTPException(
            status_code=502,
            detail="Encontrei imagens na página mas não consegui descrevê-las. Tente novamente."
        )

    return {
        "pagina_url": body.url,
        "total_imagens_encontradas": len(urls_filtradas),
        "total_descritas": len(resultados),
        "imagens": resultados,
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
