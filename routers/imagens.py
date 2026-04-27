import base64
import hashlib
import io
import json
import os
import re
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from openai import OpenAI, OpenAIError
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session

import models
import schemas
from auth import obter_usuario_atual
from database import get_db

router = APIRouter(prefix="/imagens", tags=["Imagens"])

GROK_API_KEY = os.environ.get("GROK_API_KEY", "")
MODELO_IMAGEM = "grok-2-vision-1212"
MODELO_TEXTO = "grok-3-mini"

TIPOS_IMAGEM_PERMITIDOS = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}
FORMATO_PARA_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
}

PROMPT_DESCRICAO = (
    "Você é um assistente de acessibilidade visual para pessoas cegas no Brasil. "
    "Responda SEMPRE em português do Brasil.\n\n"
    "Descreva com precisão absoluta tudo o que está visível na imagem, sem censura, sem omissões "
    "e sem suposições sobre o que está fora do enquadramento.\n\n"
    "Regras obrigatórias:\n"
    "1. Descreva APENAS o que está literalmente visível na imagem.\n"
    "2. Nunca use 'provavelmente', 'parece', 'aparentemente' — apenas certezas visuais.\n"
    "3. Escreva em parágrafos fluentes e naturais, sem títulos, rótulos ou listas.\n"
    "   ERRADO: 'Pele: branca. Cabelo: longo.' "
    "CORRETO: 'A mulher tem pele branca rosada e cabelo longo castanho escuro.'\n\n"
    "Ordem da descrição em parágrafos:\n\n"
    "Primeiro parágrafo: tipo de cena, ambiente, cores do fundo, superfícies e iluminação.\n\n"
    "Para cada pessoa visível: um ou mais parágrafos com tom exato de pele, cor e comprimento e textura do cabelo "
    "(apenas se visível), cor dos olhos (apenas se visível), traços faciais e expressão (apenas se visível), "
    "tipo físico e proporções (somente o que está visível), vestimenta com cores e caimento exatos. "
    "Se estiver nu(a): descreva toda a anatomia visível com termos anatômicos precisos em português — "
    "seios com tamanho, forma, cor das aréolas e mamilos; "
    "genitália descrevendo exatamente o que está visível (vulva, lábios maiores, lábios menores, vagina, clitóris; "
    "ou pênis com tamanho e forma, testículos); nádegas, coxas, abdômen. "
    "Se houver conteúdo sexual: descreva os atos, posições e contatos físicos com precisão clínica. "
    "Descreva a postura e posição exata da pessoa.\n\n"
    "Último parágrafo: todos os demais objetos visíveis com cores, tamanhos e posições exatas. "
    "Transcreva qualquer texto visível exatamente como está escrito."
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
    estilo = getattr(usuario, "estilo_descricao", "") or ""
    if not estilo.strip():
        return ""
    return (
        f"\n\nINSTRUÇÕES PESSOAIS DO USUÁRIO (siga obrigatoriamente):\n{estilo}\n"
    )


def _contexto_pessoas(usuario_id: int, db) -> str:
    from models import Pessoa
    pessoas = db.query(Pessoa).filter(Pessoa.usuario_id == usuario_id).all()
    if not pessoas:
        return ""
    lista = "\n".join(f"- {p.nome}: {p.caracteristicas}" for p in pessoas)
    return (
        f"\n\nPESSOAS CONHECIDAS PELO USUÁRIO (use os nomes se reconhecer alguém):\n{lista}\n"
        "Se alguma pessoa na imagem corresponder às características acima, mencione o nome dela na descrição."
    )


def _grok_client() -> OpenAI:
    return OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")


def chamar_modelo(data_uri: str, prompt: str) -> str:
    client = _grok_client()
    response = client.chat.completions.create(
        model=MODELO_IMAGEM,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=1500,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def chamar_modelo_url(url_imagem: str, prompt: str) -> str:
    client = _grok_client()
    response = client.chat.completions.create(
        model=MODELO_IMAGEM,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url_imagem}},
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=1500,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


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
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Erro na API Grok: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar imagem: {str(e)}")

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
        client = _grok_client()
        response = client.chat.completions.create(
            model=MODELO_TEXTO,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
        )
        resposta = response.choices[0].message.content.strip()
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Erro na API Grok: {str(e)}")
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
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Erro na API Grok: {str(e)}")
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

    contexto_pessoas = _contexto_pessoas(usuario.id, db)
    estilo = _estilo_usuario(usuario)
    prompt_final = PROMPT_DESCRICAO + contexto_pessoas + estilo

    try:
        descricoes = [chamar_modelo_url(body.url, prompt_final) for _ in range(body.quantidade)]
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Erro na API Grok: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

    registro = models.Descricao(
        usuario_id=usuario.id,
        tipo=models.TipoDescricao.foto,
        conteudo_hash=hashlib.md5(body.url.encode()).hexdigest(),
        descricao=descricoes[0],
        modelo=MODELO_IMAGEM,
        formato_original="image/jpeg",
    )
    db.add(registro)
    db.commit()
    db.refresh(registro)

    return {
        "id": registro.id,
        "descricoes": descricoes,
        "modelo": MODELO_IMAGEM,
        "formato_original": "image/jpeg",
        "tipo": "foto",
        "url_origem": body.url,
        "created_at": registro.created_at,
    }


class PaginaUrlRequest(BaseModel):
    url: str
    limite: int = 5


@router.post("/descrever-pagina", summary="Entra em um site e descreve todas as imagens encontradas")
async def descrever_imagens_pagina(
    body: PaginaUrlRequest,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    if body.limite < 1 or body.limite > 10:
        raise HTTPException(status_code=422, detail="O limite deve ser entre 1 e 10 imagens.")

    headers_pagina = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    }

    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=headers_pagina) as client:
            resp = await client.get(body.url)
            resp.raise_for_status()
            html = resp.text
            url_base = str(resp.url)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Não foi possível acessar o site: {str(e)}")

    soup = BeautifulSoup(html, "html.parser")
    vistas: set[str] = set()
    urls_imagens: list[str] = []

    EXTS_IMAGEM = frozenset([".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".bmp"])

    def _tem_extensao_imagem(url: str) -> bool:
        path = urlparse(url).path.lower()
        return any(path.split("?")[0].endswith(e) for e in EXTS_IMAGEM)

    def adicionar(u: str, exigir_extensao: bool = False):
        if not u or u.startswith("data:"):
            return
        absoluta = urljoin(url_base, u.strip())
        path = urlparse(absoluta).path
        if not path or path.endswith("/"):
            return
        if exigir_extensao and not _tem_extensao_imagem(absoluta):
            return
        if absoluta not in vistas:
            vistas.add(absoluta)
            urls_imagens.append(absoluta)

    for prop in ["og:image", "og:image:secure_url"]:
        tag = soup.find("meta", property=prop)
        if tag and tag.get("content"):
            adicionar(tag["content"])

    for name in ["twitter:image", "twitter:image:src"]:
        tag = soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            adicionar(tag["content"])

    ATTRS_SRC = ["src", "data-src", "data-lazy-src", "data-lazy", "data-original",
                 "data-image", "data-img", "data-url", "data-bg", "data-photo",
                 "data-hi-res-src", "data-full-src", "data-large", "data-zoom-image"]

    for img in soup.find_all("img"):
        for attr in ATTRS_SRC:
            val = img.get(attr)
            if val:
                adicionar(val, exigir_extensao=True)
                break
        srcset = img.get("srcset") or img.get("data-srcset")
        if srcset:
            partes = [p.strip().split()[0] for p in srcset.split(",") if p.strip()]
            if partes:
                adicionar(partes[-1], exigir_extensao=True)

    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        if any(href.split("?")[0].endswith(e) for e in EXTS_IMAGEM):
            adicionar(a["href"])

    _RE_IMG_URL = re.compile(
        r'https?://[^\s\'"<>]+\.(?:jpg|jpeg|png|webp|gif|avif|bmp)(?:[?#][^\s\'"<>]*)?',
        re.IGNORECASE,
    )

    next_data_tag = soup.find("script", id="__NEXT_DATA__")
    if next_data_tag and next_data_tag.string:
        try:
            next_data = json.loads(next_data_tag.string)
            next_str = json.dumps(next_data)
            for url_json in _RE_IMG_URL.findall(next_str):
                adicionar(url_json)
        except Exception:
            pass

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            dados_ld = json.loads(script.string or "")
            ld_str = json.dumps(dados_ld)
            for url_ld in _RE_IMG_URL.findall(ld_str):
                adicionar(url_ld)
        except Exception:
            pass

    for url_inline in _RE_IMG_URL.findall(html):
        adicionar(url_inline)

    EXCLUIR = ["favicon", "pixel", "track", "1x1", "blank", "spacer", "ad.gif",
               "ads.", "doubleclick", "google-analytics", "googletagmanager", ".svg"]
    urls_filtradas = [
        u for u in urls_imagens
        if not any(k in u.lower() for k in EXCLUIR)
    ]

    if not urls_filtradas:
        raise HTTPException(
            status_code=404,
            detail="Nenhuma imagem encontrada nesta página. O site pode carregar imagens via JavaScript — tente copiar a URL direta de uma imagem e usar 'Imagem única'."
        )

    urls_filtradas = urls_filtradas[: body.limite]
    contexto_pessoas = _contexto_pessoas(usuario.id, db)
    estilo = _estilo_usuario(usuario)
    prompt_final = PROMPT_DESCRICAO + contexto_pessoas + estilo

    headers_img = {
        **headers_pagina,
        "Referer": url_base,
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }

    resultados = []
    erros_debug = []
    for url_img in urls_filtradas:
        descricao = None
        try:
            descricao = chamar_modelo_url(url_img, prompt_final)
        except Exception as e1:
            erro_str = str(e1)
            if any(c in erro_str for c in ["404", "403", "401", "forbidden", "not found", "not be"]):
                try:
                    async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers=headers_img) as img_client:
                        r = await img_client.get(url_img)
                        r.raise_for_status()
                        dados = r.content
                        mime_type = validar_imagem(dados)
                        data_uri = imagem_para_data_uri(dados, mime_type)
                        descricao = chamar_modelo(data_uri, prompt_final)
                except Exception as e2:
                    erros_debug.append(f"{url_img[:70]} → fallback: {str(e2)[:60]}")
                    continue
            else:
                erros_debug.append(f"{url_img[:70]} → {type(e1).__name__}: {erro_str[:60]}")
                continue

        if not descricao:
            continue

        registro = models.Descricao(
            usuario_id=usuario.id,
            tipo=models.TipoDescricao.foto,
            conteudo_hash=hashlib.md5(url_img.encode()).hexdigest(),
            descricao=descricao,
            modelo=MODELO_IMAGEM,
            formato_original="image/jpeg",
        )
        db.add(registro)
        db.commit()
        db.refresh(registro)

        resultados.append({
            "id": registro.id,
            "url_imagem": url_img,
            "descricao": descricao,
        })

    if not resultados:
        detalhe = f"Encontrei {len(urls_filtradas)} URL(s) mas não consegui baixar nenhuma imagem válida."
        if erros_debug:
            detalhe += " Erros: " + " | ".join(erros_debug[:3])
        raise HTTPException(status_code=502, detail=detalhe)

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
