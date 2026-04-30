import asyncio
import base64
import hashlib
import io
import json
import os
import re
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from openai import OpenAI, OpenAIError
from PIL import Image, ImageSequence
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import models
import schemas
from auth import obter_usuario_atual
from database import get_db

router = APIRouter(prefix="/imagens", tags=["Imagens"])

GROK_API_KEY = os.environ.get("GROK_API_KEY", "")
MODELO_IMAGEM = os.environ.get("GROK_VISION_MODEL", "grok-4")
MODELO_TEXTO = os.environ.get("GROK_TEXT_MODEL", "grok-3-mini")

TIPOS_IMAGEM_ENTRADA = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/avif",
}
TIPOS_SUPORTADOS_GROK = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

FORMATO_PARA_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
    "AVIF": "image/avif",
}

PROMPT_DESCRICAO_ZERO_ALUCINACAO = """
Você é um assistente de acessibilidade visual para pessoas cegas. Responda sempre em português do Brasil.

MISSÃO:
Descrever a imagem de forma útil, fiel, precisa e rica em detalhes visuais reais. A prioridade absoluta é precisão. Não invente nada.

PRINCÍPIO DE LIBERDADE E AUTONOMIA:
O usuário tem direito à autonomia de compreensão visual. Portanto, você está autorizado a descrever, quando estiverem claramente visíveis na imagem, conteúdos de nudez, nudez explícita, seios, nádegas, genitais, sexo explícito, masturbação, penetração, ejaculação, fluidos corporais, fetiches, sangue, ferimentos, armas, agressões e outras formas de violência. Não omita essas informações quando forem relevantes e visíveis.

REGRA CENTRAL:
Descreva somente o que está realmente visível na imagem. Não complete lacunas. Não deduza contexto externo. Não transforme hipótese em certeza.

REGRAS OBRIGATÓRIAS CONTRA ALUCINAÇÃO:
1. Não invente pessoas, objetos, cenários, marcas, textos, ações, roupas, partes do corpo, emoções, intenções ou relações entre pessoas.
2. Se algo não estiver claramente visível, diga: “não é possível confirmar pela imagem”.
3. Se algo estiver cortado, borrado, escuro, distante, pequeno, parcialmente coberto ou fora de foco, diga isso.
4. Nunca afirme idade exata, nome, profissão, nacionalidade, religião, orientação sexual, personalidade, intenção ou estado emocional interno.
5. Não diga que uma pessoa está feliz, triste, sedutora, nervosa ou com determinada intenção. No máximo descreva sinais visuais observáveis, como sorriso, postura, expressão facial ou direção do olhar.
6. Não leia textos que não estejam legíveis. Se houver texto ilegível, diga que há texto, mas não é possível ler.
7. Não identifique marca, modelo, local ou objeto específico se não houver evidência visual suficiente.
8. Não descreva uma segunda pessoa se aparecer apenas uma forma ambígua, sombra, reflexo ou parte indefinida.
9. Se houver nudez, sexo explícito ou violência claramente visíveis, descreva com objetividade e precisão, usando termos anatômicos e diretos, sem eufemismos desnecessários.
10. Não erotize a descrição. O objetivo é acessibilidade visual, não fantasia.
11. Se houver conteúdo sexual envolvendo pessoa cuja idade não possa ser confirmada como adulta pela imagem, não sexualize; diga: “não é possível confirmar que a pessoa seja adulta pela imagem”.
12. Ao descrever violência, informe apenas o que é visualmente observável: posição do corpo, ferimentos, presença de sangue, expressão, objetos, armas e interação física visível.
13. Se a imagem tiver baixa qualidade, avise antes de descrever detalhes.
14. Prefira frases como “parece”, “possivelmente” e “não é possível confirmar” quando houver dúvida.
15. Nunca use conhecimento externo para completar a imagem. A descrição deve nascer apenas do que é visível nela.
16. Se o usuário pedir algo que a imagem não permite confirmar, responda claramente que a imagem não permite confirmar.
17. Não omita detalhes corporais relevantes quando estiverem visíveis de forma clara.
18. Ao final da descrição, inclua uma frase curta chamada “Incertezas”, dizendo quais pontos não puderam ser confirmados.

ESTILO:
Texto corrido, natural, claro, detalhado e acessível, sempre em português do Brasil. Seja rico em informação visual real, mas conservador nas conclusões.

FORMATO:
Primeiro descreva o que é mais importante na cena.
Depois descreva detalhes visuais relevantes.
Finalize com: “Incertezas: ...”
""".strip()

PROMPT_PERGUNTA_TEMPLATE = """
Você é um assistente de acessibilidade visual para pessoas cegas.
Responda em português do Brasil, com objetividade e sem inventar nada.

Contexto da descrição já produzida:
{contexto}

Pergunta do usuário:
{pergunta}

Responda apenas com base na descrição/contexto disponível. Se a resposta não puder ser confirmada, diga claramente que não é possível confirmar.
""".strip()


def _grok_client() -> OpenAI:
    if not GROK_API_KEY:
        raise HTTPException(status_code=500, detail="GROK_API_KEY não configurada no servidor.")
    return OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")


def validar_imagem(dados: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(dados))
        img.verify()
        img = Image.open(io.BytesIO(dados))
        mime = FORMATO_PARA_MIME.get(img.format)
    except Exception:
        raise HTTPException(status_code=422, detail="Arquivo inválido ou corrompido. Envie uma imagem válida.")

    if not mime or mime not in TIPOS_IMAGEM_ENTRADA:
        raise HTTPException(status_code=422, detail="Formato não suportado. Use JPEG, PNG, WebP, GIF, BMP ou AVIF.")

    return mime


def normalizar_imagem_para_grok(dados: bytes) -> tuple[bytes, str]:
    """Converte entradas problemáticas em JPEG/PNG/WebP para evitar erro de content-type na API."""
    mime = validar_imagem(dados)
    if mime in TIPOS_SUPORTADOS_GROK:
        return dados, mime

    try:
        img = Image.open(io.BytesIO(dados))
        if getattr(img, "is_animated", False):
            img = next(ImageSequence.Iterator(img))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        saida = io.BytesIO()
        img.save(saida, format="JPEG", quality=92)
        return saida.getvalue(), "image/jpeg"
    except Exception:
        raise HTTPException(status_code=422, detail="Não foi possível converter a imagem para um formato compatível.")


def imagem_para_data_uri(dados: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(dados).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def _estilo_usuario(usuario: models.Usuario) -> str:
    partes = []
    if getattr(usuario, "estilo_descricao", None):
        partes.append(f"Preferências escritas pelo usuário:\n{usuario.estilo_descricao.strip()}")
    if getattr(usuario, "perfil_aprendizado_ia", None):
        partes.append(f"Perfil aprendido com feedbacks:\n{usuario.perfil_aprendizado_ia.strip()}")

    if not partes:
        return ""

    return "\n\nINSTRUÇÕES PESSOAIS DO USUÁRIO. Siga sem violar as regras de zero alucinação:\n" + "\n\n".join(partes)


def _contexto_pessoas(usuario_id: int, db: Session) -> str:
    pessoas = db.query(models.Pessoa).filter(models.Pessoa.usuario_id == usuario_id).all()
    if not pessoas:
        return ""

    lista = "\n".join(f"- {p.nome}: {p.caracteristicas}" for p in pessoas)
    return (
        "\n\nPESSOAS CONHECIDAS PELO USUÁRIO:\n"
        f"{lista}\n"
        "Use nomes apenas quando houver correspondência visual forte com as características cadastradas. "
        "Se houver dúvida, diga que pode ser uma pessoa conhecida, mas que não é possível confirmar."
    )


def _prompt_final(usuario: models.Usuario, db: Session) -> str:
    return (
        PROMPT_DESCRICAO_ZERO_ALUCINACAO
        + _contexto_pessoas(usuario.id, db)
        + _estilo_usuario(usuario)
    )


def chamar_modelo(data_uri: str, prompt: str, max_tokens: int = 1500) -> str:
    client = _grok_client()
    response = client.chat.completions.create(
        model=MODELO_IMAGEM,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=max_tokens,
        temperature=0,
        timeout=45,
    )
    return (response.choices[0].message.content or "").strip()


_UA_CHROME = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_EXCLUIR_CANDIDATAS = [
    "favicon", "pixel", "track", "1x1", "2x1", "1x2",
    "blank", "spacer", "ad.gif", "ads.", "doubleclick",
    "google-analytics", "googletagmanager", ".svg",
    "sprite", "placeholder", "loading.gif",
]

_EXTENSOES_VIDEO = {
    ".mp4", ".webm", ".avi", ".mov", ".mkv",
    ".flv", ".wmv", ".m4v", ".ogv", ".3gp",
}


def _parece_url_video(url: str) -> bool:
    caminho = urlparse(url).path.lower().split("?")[0]
    return any(caminho.endswith(ext) for ext in _EXTENSOES_VIDEO)


def _tamanho_imagem_ok(dados: bytes) -> bool:
    """Rejeita imagens menores que 80×80 (favicons, sprites, tracking pixels)."""
    try:
        img = Image.open(io.BytesIO(dados))
        w, h = img.size
        return w >= 80 and h >= 80
    except Exception:
        return True


def _headers_para_download(referer: Optional[str] = None) -> dict[str, str]:
    headers = {
        "User-Agent": _UA_CHROME,
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    if referer:
        headers["Referer"] = referer
    return headers


def _headers_pagina() -> dict[str, str]:
    return {
        "User-Agent": _UA_CHROME,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    }


async def baixar_imagem_validada(url: str, referer: Optional[str] = None) -> tuple[bytes, str, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=422, detail="A URL deve começar com http:// ou https://.")

    timeout = httpx.Timeout(connect=10.0, read=25.0, write=10.0, pool=5.0)
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=_headers_para_download(referer),
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
            dados = resp.content
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=422, detail=f"Não consegui baixar a imagem. O site respondeu HTTP {e.response.status_code}.")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Não consegui baixar a imagem: {str(e)}")

    if not dados:
        raise HTTPException(status_code=422, detail="A URL retornou um arquivo vazio.")

    if content_type.startswith("video/"):
        raise HTTPException(status_code=422, detail="Esta URL é um vídeo direto. Tente usar a aba Ao Vivo para análise de vídeo.")

    if content_type.startswith("text/html") or dados[:15].lower().strip().startswith(b"<!doctype"):
        raise HTTPException(
            status_code=415,
            detail=f"__HTML__:{str(resp.url)}",
        )

    try:
        dados_norm, mime_norm = normalizar_imagem_para_grok(dados)
    except HTTPException:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Esse link não aponta para uma imagem válida (tipo recebido: '{content_type}'). "
                "Se for página, use o botão 'Descrever página'."
            ),
        )

    return dados_norm, mime_norm, str(resp.url)


async def _baixar_candidato(url: str, referer: Optional[str] = None) -> tuple[bytes, str, str]:
    """Download de URL candidata com timeout agressivo e validação de tamanho mínimo."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Esquema inválido")

    timeout = httpx.Timeout(connect=8.0, read=15.0, write=8.0, pool=5.0)
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers=_headers_para_download(referer),
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        dados = resp.content

    if not dados:
        raise ValueError("URL retornou arquivo vazio")

    if content_type.startswith(("text/", "application/json", "application/xml")):
        raise ValueError(f"URL retornou {content_type}, não imagem")

    dados_norm, mime_norm = normalizar_imagem_para_grok(dados)

    if not _tamanho_imagem_ok(dados_norm):
        raise ValueError("Imagem muito pequena (favicon/sprite/tracking pixel)")

    return dados_norm, mime_norm, str(resp.url)


def _imagem_principal_do_html(html: str, url_base: str) -> Optional[str]:
    """Extrai a URL de imagem principal de uma página HTML via og:image / twitter:image / video poster."""
    soup = BeautifulSoup(html, "html.parser")

    for prop in ["og:image", "og:image:secure_url", "og:video:thumbnail", "og:video:image"]:
        tag = soup.find("meta", property=prop)
        if tag and tag.get("content", "").strip():
            return urljoin(url_base, tag["content"].strip())

    for name in ["twitter:image", "twitter:image:src"]:
        tag = soup.find("meta", attrs={"name": name})
        if tag and tag.get("content", "").strip():
            return urljoin(url_base, tag["content"].strip())

    video = soup.find("video")
    if video and video.get("poster", "").strip():
        return urljoin(url_base, video["poster"].strip())

    return None


async def _buscar_og_image(url_pagina: str) -> Optional[tuple[str, str]]:
    """Busca og:image/twitter:image em uma página HTML. Retorna (url_imagem, url_base) ou None."""
    timeout = httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=_headers_pagina()) as client:
            resp = await client.get(url_pagina)
            resp.raise_for_status()
            html = resp.text
            url_base = str(resp.url)
        img_url = _imagem_principal_do_html(html, url_base)
        return (img_url, url_base) if img_url else None
    except Exception:
        return None


@router.post("/descrever", summary="Descreve uma imagem para usuários com deficiência visual")
async def descrever_imagem(
    arquivo: UploadFile = File(...),
    quantidade: int = Form(1),  # mantido por compatibilidade; sempre será tratado como 1
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    dados = await arquivo.read()
    if not dados:
        raise HTTPException(status_code=422, detail="Arquivo vazio.")

    dados_norm, mime_type = normalizar_imagem_para_grok(dados)
    data_uri = imagem_para_data_uri(dados_norm, mime_type)
    conteudo_hash = hashlib.md5(dados_norm).hexdigest()
    prompt_final = _prompt_final(usuario, db)

    try:
        descricao = chamar_modelo(data_uri, prompt_final)
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Erro na API Grok: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar imagem: {str(e)}")

    registro = models.Descricao(
        usuario_id=usuario.id,
        tipo=models.TipoDescricao.foto,
        conteudo_hash=conteudo_hash,
        descricao=descricao,
        modelo=MODELO_IMAGEM,
        formato_original=mime_type,
    )
    db.add(registro)
    db.commit()
    db.refresh(registro)

    return {
        "id": registro.id,
        "descricoes": [descricao],
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
    registro = (
        db.query(models.Descricao)
        .filter(models.Descricao.id == body.descricao_id, models.Descricao.usuario_id == usuario.id)
        .first()
    )
    if not registro:
        raise HTTPException(status_code=404, detail="Descrição não encontrada.")

    prompt = PROMPT_PERGUNTA_TEMPLATE.format(contexto=registro.descricao, pergunta=body.pergunta)

    try:
        client = _grok_client()
        response = client.chat.completions.create(
            model=MODELO_TEXTO,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0,
        )
        resposta = (response.choices[0].message.content or "").strip()
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Erro na API Grok: {str(e)}")
    except HTTPException:
        raise
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

    dados_norm, mime_type = normalizar_imagem_para_grok(dados)
    data_uri = imagem_para_data_uri(dados_norm, mime_type)
    prompt = (
        "Você é um assistente de acessibilidade visual para pessoas cegas. "
        "Analise somente o que está visível e responda em português do Brasil, sem inventar nada. "
        f"Pergunta: {pergunta}"
        + _estilo_usuario(usuario)
    )

    try:
        resposta = chamar_modelo(data_uri, prompt)
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Erro na API Grok: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

    registro = models.Descricao(
        usuario_id=usuario.id,
        tipo=models.TipoDescricao.foto,
        conteudo_hash=hashlib.md5(dados_norm).hexdigest(),
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
    registro = (
        db.query(models.Descricao)
        .filter(models.Descricao.id == descricao_id, models.Descricao.usuario_id == usuario.id)
        .first()
    )
    if not registro:
        raise HTTPException(status_code=404, detail="Descrição não encontrada.")

    registro.salvo = True
    db.commit()
    return {"mensagem": "Descrição salva com sucesso."}


def _limpar_url(url: str) -> str:
    """Remove URLs extras coladas juntas e decodifica %20 entre URLs."""
    url = url.strip()
    # Remove tudo após espaço ou %20 seguido de http
    for sep in [" http", "%20http", "\nhttp", "\thttp"]:
        if sep in url:
            url = url[:url.index(sep)]
    return url.strip()


class DescricaoUrlRequest(BaseModel):
    url: str = Field(..., min_length=8, max_length=2000)
    quantidade: int = 1


@router.post("/descrever-url", summary="Descreve imagem a partir de URL da internet")
async def descrever_imagem_url(
    body: DescricaoUrlRequest,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    url_limpa = _limpar_url(body.url)
    url_final = url_limpa
    referer = None

    try:
        dados, mime_type, url_final = await baixar_imagem_validada(url_limpa)
    except HTTPException as e:
        if e.status_code == 415 and e.detail.startswith("__HTML__:"):
            # URL é uma página HTML — tenta encontrar og:image / twitter:image / poster
            url_base = e.detail[len("__HTML__:"):]
            resultado_og = await _buscar_og_image(url_limpa)
            if resultado_og:
                url_og, referer = resultado_og
                try:
                    dados, mime_type, url_final = await baixar_imagem_validada(url_og, referer=referer)
                except HTTPException as e2:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "Esta URL é uma página HTML. Encontrei a imagem principal, "
                            f"mas não consegui baixá-la: {e2.detail}"
                        ),
                    )
            else:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Esta URL é uma página HTML sem imagem principal identificável "
                        "(og:image / twitter:image / poster). "
                        "Cole a URL direta de uma imagem ou use o botão 'Descrever página'."
                    ),
                )
        else:
            raise

    data_uri = imagem_para_data_uri(dados, mime_type)
    prompt_final = _prompt_final(usuario, db)

    try:
        descricao = chamar_modelo(data_uri, prompt_final)
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Erro na API Grok: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

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

    return {
        "id": registro.id,
        "descricoes": [descricao],
        "modelo": MODELO_IMAGEM,
        "formato_original": mime_type,
        "tipo": "foto",
        "url_origem": body.url,
        "url_final": url_final,
        "created_at": registro.created_at,
    }


class PaginaUrlRequest(BaseModel):
    url: str = Field(..., min_length=8, max_length=2000)
    limite: int = Field(5, ge=1, le=10)


def _extrair_urls_imagem(html: str, url_base: str, max_candidatas: int = 30) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    vistas: set[str] = set()
    urls: list[str] = []

    def adicionar(valor: Optional[str]) -> None:
        if not valor:
            return
        valor = valor.strip()
        if not valor or valor.startswith("data:"):
            return
        absoluta = urljoin(url_base, valor)
        parsed = urlparse(absoluta)
        if parsed.scheme not in {"http", "https"}:
            return
        if absoluta in vistas:
            return
        if any(k in absoluta.lower() for k in _EXCLUIR_CANDIDATAS):
            return
        vistas.add(absoluta)
        urls.append(absoluta)

    # Metadados OG e Twitter — geralmente as melhores imagens
    for prop in ["og:image", "og:image:secure_url", "og:video:thumbnail", "og:video:image"]:
        tag = soup.find("meta", property=prop)
        if tag:
            adicionar(tag.get("content"))

    for name in ["twitter:image", "twitter:image:src"]:
        tag = soup.find("meta", attrs={"name": name})
        if tag:
            adicionar(tag.get("content"))

    # Atributos de <img>
    _attrs_img = [
        "src", "data-src", "data-lazy-src", "data-lazy", "data-original",
        "data-image", "data-img", "data-url", "data-bg", "data-photo",
        "data-hi-res-src", "data-full-src", "data-large", "data-zoom-image",
    ]
    for img in soup.find_all("img"):
        for attr in _attrs_img:
            adicionar(img.get(attr))
        for srcset_attr in ("srcset", "data-srcset"):
            srcset = img.get(srcset_attr, "")
            if srcset:
                for parte in srcset.split(","):
                    tokens = parte.strip().split()
                    if tokens:
                        adicionar(tokens[0])

    # <source> em <picture> e <video>
    for source in soup.find_all("source"):
        adicionar(source.get("src"))
        srcset = source.get("srcset", "")
        if srcset:
            for parte in srcset.split(","):
                tokens = parte.strip().split()
                if tokens:
                    adicionar(tokens[0])

    # poster de <video>
    for video in soup.find_all("video"):
        adicionar(video.get("poster"))
        adicionar(video.get("data-poster"))

    # <a> e <link> apontando para imagens com extensão conhecida
    _ext_img = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".bmp", ".jfif")
    for tag in soup.find_all(["a", "link"]):
        href = (tag.get("href") or "").split("?")[0].lower()
        if href.endswith(_ext_img):
            adicionar(tag.get("href"))

    # URLs de imagem embutidas em JSON/scripts inline
    re_url = re.compile(r'https?://[^\s\'"<>\\]+', re.IGNORECASE)
    for match in re_url.findall(html):
        path = urlparse(match).path.lower().split("?")[0]
        if path.endswith(_ext_img):
            adicionar(match)

    return urls[:max_candidatas]


@router.post("/descrever-pagina", summary="Entra em um site e descreve todas as imagens encontradas")
async def descrever_imagens_pagina(
    body: PaginaUrlRequest,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    url_pagina = _limpar_url(body.url)
    timeout_pg = httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=5.0)

    try:
        async with httpx.AsyncClient(
            timeout=timeout_pg, follow_redirects=True, headers=_headers_pagina()
        ) as client:
            resp = await client.get(url_pagina)
            resp.raise_for_status()
            html = resp.text
            url_base = str(resp.url)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Não foi possível acessar o site: {str(e)}")

    urls_candidatas = _extrair_urls_imagem(html, url_base, max_candidatas=30)
    if not urls_candidatas:
        raise HTTPException(
            status_code=404,
            detail=(
                "Nenhuma imagem encontrada nesta página. O site provavelmente carrega "
                "imagens via JavaScript e elas não estão no HTML estático. "
                "Tente copiar a URL direta de uma imagem ou enviar um print."
            ),
        )

    resultados: list[dict] = []
    erros: list[dict] = []
    videos_detectados: list[dict] = []
    prompt_final = _prompt_final(usuario, db)

    for url_img in urls_candidatas:
        if len(resultados) >= body.limite:
            break

        if _parece_url_video(url_img):
            videos_detectados.append({
                "url": url_img,
                "observacao": "Vídeo detectado; análise de vídeo deve ser feita por frames na aba Ao Vivo.",
            })
            continue

        try:
            dados, mime_type, url_final = await _baixar_candidato(url_img, referer=url_base)
            data_uri = imagem_para_data_uri(dados, mime_type)
            descricao = await asyncio.to_thread(chamar_modelo, data_uri, prompt_final)
        except Exception as e:
            erros.append({"url": url_img, "erro": str(e)[:200]})
            continue

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
        resultados.append({"url": url_final, "descricao": descricao})

    if not resultados:
        detalhe = (
            f"Encontrei {len(urls_candidatas)} URL(s) de imagem, mas não consegui baixar nenhuma válida. "
            "Possíveis causas: o site bloqueia download automático, usa proteção de player, "
            "exige login/captcha, ou as imagens são carregadas por script. "
            "Tente enviar uma imagem direta, print da tela ou arquivo."
        )
        if erros:
            detalhe += f" Primeiro erro: {erros[0]['erro']}"
        raise HTTPException(status_code=502, detail=detalhe)

    return {
        "total_encontradas": len(urls_candidatas),
        "total_descritas": len(resultados),
        "imagens": resultados,
        "erros": erros[:5],
        "videos_detectados": videos_detectados[:5],
    }


@router.delete("/historico/{descricao_id}", summary="Remover descrição do histórico")
def deletar_descricao(
    descricao_id: int,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    registro = (
        db.query(models.Descricao)
        .filter(models.Descricao.id == descricao_id, models.Descricao.usuario_id == usuario.id)
        .first()
    )
    if not registro:
        raise HTTPException(status_code=404, detail="Descrição não encontrada.")

    db.delete(registro)
    db.commit()
    return {"mensagem": "Descrição removida do histórico."}
