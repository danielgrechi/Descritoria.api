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

_EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif", ".jfif", ".tiff", ".tif"}
_EXTENSOES_VIDEO = {".mp4", ".webm", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".ogv", ".3gp"}

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
    )
    return (response.choices[0].message.content or "").strip()


def _headers_para_download(referer: Optional[str] = None) -> dict[str, str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "image",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "cross-site",
    }
    if referer:
        headers["Referer"] = referer
    return headers


def _parece_url_imagem(url: str) -> bool:
    caminho = urlparse(url).path.lower()
    caminho_sem_qs = caminho.split("?")[0]
    return any(caminho_sem_qs.endswith(ext) for ext in _EXTENSOES_IMAGEM)


def _parece_url_video(url: str) -> bool:
    caminho = urlparse(url).path.lower().split("?")[0]
    return any(caminho.endswith(ext) for ext in _EXTENSOES_VIDEO)


def _extrair_urls_srcset(srcset: str) -> list[str]:
    """Extrai URLs de um atributo srcset; retorna todas (sem filtrar por resolução)."""
    urls = []
    for parte in srcset.split(","):
        parte = parte.strip()
        if not parte:
            continue
        tokens = parte.split()
        if tokens:
            urls.append(tokens[0])
    return urls


async def baixar_imagem_validada(url: str, referer: Optional[str] = None) -> tuple[bytes, str, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=422, detail="A URL deve começar com http:// ou https://.")

    if _parece_url_video(url):
        raise HTTPException(
            status_code=422,
            detail="Esta URL aponta para um vídeo direto. Esta aba suporta somente imagens. Para análise de vídeo, use a aba Câmera Ao Vivo.",
        )

    try:
        async with httpx.AsyncClient(
            timeout=30,
            follow_redirects=True,
            headers=_headers_para_download(referer),
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
            dados = resp.content
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Não consegui baixar a imagem. O site respondeu HTTP {e.response.status_code}.",
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Não consegui baixar a imagem: {str(e)}")

    if not dados:
        raise HTTPException(status_code=422, detail="A URL retornou um arquivo vazio.")

    if content_type.startswith("video/"):
        raise HTTPException(
            status_code=422,
            detail="Esta URL é um vídeo direto. Esta aba suporta somente imagens. Para análise de vídeo, use a aba Câmera Ao Vivo.",
        )

    if content_type in {"text/html", "text/xml", "application/xhtml+xml"} or dados[:15].lower().strip().startswith(b"<!doctype"):
        raise HTTPException(
            status_code=422,
            detail="Esta URL retornou uma página HTML, não uma imagem. Cole a URL direta da imagem (terminando em .jpg, .png etc.) ou use o botão 'Descrever página'.",
        )

    try:
        dados_norm, mime_norm = normalizar_imagem_para_grok(dados)
    except HTTPException:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Esse link não aponta para uma imagem válida (tipo recebido: '{content_type}'). "
                "Se for imagem de uma página, use o botão 'Descrever página'."
            ),
        )

    return dados_norm, mime_norm, str(resp.url)


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


class DescricaoUrlRequest(BaseModel):
    url: str = Field(..., min_length=8, max_length=2000)
    quantidade: int = 1


@router.post("/descrever-url", summary="Descreve imagem a partir de URL da internet")
async def descrever_imagem_url(
    body: DescricaoUrlRequest,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    dados, mime_type, url_final = await baixar_imagem_validada(body.url)
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


@router.post("/diagnostico-pagina", summary="Diagnóstico: mostra o que o servidor consegue extrair de uma página")
async def diagnostico_pagina(
    body: PaginaUrlRequest,
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    headers_pagina = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers=headers_pagina) as client:
            resp = await client.get(body.url)
            status_code = resp.status_code
            content_type = resp.headers.get("content-type", "")
            html = resp.text
            url_base = str(resp.url)
    except Exception as e:
        return {"erro_conexao": str(e)}

    tem_next_data = "__NEXT_DATA__" in html
    tem_nuxt = "__NUXT__" in html
    urls_json = _extrair_json_embutido(html)
    urls_todas = _coletar_urls_imagens_da_pagina(html, url_base)

    return {
        "status_http": status_code,
        "content_type": content_type,
        "html_tamanho": len(html),
        "html_inicio": html[:500],
        "tem_next_data": tem_next_data,
        "tem_nuxt": tem_nuxt,
        "urls_do_json_embutido": urls_json[:10],
        "total_urls_encontradas": len(urls_todas),
        "primeiras_urls": urls_todas[:10],
    }


def _extrair_json_embutido(html: str) -> list[str]:
    """Extrai URLs de imagem de blocos JSON embutidos no HTML (Next.js, Nuxt, etc.)."""
    urls: list[str] = []
    re_url = re.compile(r'https?://[^\s\'"<>\\,\]}\)]+', re.IGNORECASE)

    # __NEXT_DATA__ (Next.js), __NUXT__ (Nuxt), window.__STATE__ e similares
    script_re = re.compile(
        r'<script[^>]*(?:id=["\']__NEXT_DATA__["\']|type=["\']application/json["\'])[^>]*>(.*?)</script>',
        re.DOTALL | re.IGNORECASE,
    )
    for bloco in script_re.findall(html):
        for u in re_url.findall(bloco):
            if _parece_url_imagem(u):
                urls.append(u)

    # qualquer <script> com JSON que contenha chaves de imagem comuns
    chaves_img = re.compile(
        r'"(?:image|img|photo|foto|thumb|thumbnail|cover|avatar|src|url|picture|preview|banner)":\s*"(https?://[^"]+)"',
        re.IGNORECASE,
    )
    for bloco in re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL | re.IGNORECASE):
        for u in chaves_img.findall(bloco):
            if _parece_url_imagem(u):
                urls.append(u)

    return urls


def _coletar_urls_imagens_da_pagina(html: str, url_base: str) -> list[str]:
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
        if absoluta not in vistas:
            vistas.add(absoluta)
            urls.append(absoluta)

    # JSON embutido (Next.js / Nuxt / SSR) — primeiro porque tem as melhores URLs
    for u in _extrair_json_embutido(html):
        adicionar(u)

    # Metadados Open Graph e Twitter — geralmente as melhores thumbnails
    for prop in ["og:image", "og:image:secure_url", "og:video:thumbnail", "og:video:image"]:
        tag = soup.find("meta", property=prop)
        adicionar(tag.get("content") if tag else None)

    for name in ["twitter:image", "twitter:image:src", "thumbnail"]:
        tag = soup.find("meta", attrs={"name": name})
        adicionar(tag.get("content") if tag else None)

    # Atributos de <img>
    attrs_img = [
        "src", "data-src", "data-lazy-src", "data-lazy", "data-original",
        "data-image", "data-img", "data-url", "data-bg", "data-photo",
        "data-hi-res-src", "data-full-src", "data-large", "data-zoom-image",
        "data-thumb", "data-poster", "data-preview", "data-cover",
    ]
    for img in soup.find_all("img"):
        for attr in attrs_img:
            adicionar(img.get(attr))
        for srcset_attr in ["srcset", "data-srcset"]:
            srcset = img.get(srcset_attr)
            if srcset:
                for u in _extrair_urls_srcset(srcset):
                    adicionar(u)

    # Tags <source> dentro de <picture> e <video>
    for source in soup.find_all("source"):
        adicionar(source.get("src"))
        srcset = source.get("srcset")
        if srcset:
            for u in _extrair_urls_srcset(srcset):
                adicionar(u)

    # Posters de <video>
    for video in soup.find_all("video"):
        adicionar(video.get("poster"))
        adicionar(video.get("data-poster"))
        adicionar(video.get("data-thumb"))
        adicionar(video.get("data-preview"))

    # Links <a> e <link> com href apontando para imagem
    for tag in soup.find_all(["a", "link"]):
        href = tag.get("href", "")
        if href and _parece_url_imagem(href):
            adicionar(href)

    # URLs de imagem embutidas em qualquer parte do HTML (CSS inline, JSON solto)
    re_url = re.compile(r'https?://[^\s\'"<>\\]+', re.IGNORECASE)
    for match in re_url.findall(html):
        if _parece_url_imagem(match):
            adicionar(match)

    excluir = [
        "favicon", "pixel", "track", "1x1", "2x1", "1x2",
        "blank", "spacer", "ad.gif", "ads.", "doubleclick",
        "google-analytics", "googletagmanager", ".svg",
        "sprite", "placeholder", "loading.gif",
    ]
    return [u for u in urls if not any(k in u.lower() for k in excluir)]


@router.post("/descrever-pagina", summary="Entra em um site e descreve todas as imagens encontradas")
async def descrever_imagens_pagina(
    body: PaginaUrlRequest,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    headers_pagina = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers=headers_pagina) as client:
            resp = await client.get(body.url)
            resp.raise_for_status()
            html = resp.text
            url_base = str(resp.url)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Não foi possível acessar o site: {str(e)}")

    urls_candidatas = _coletar_urls_imagens_da_pagina(html, url_base)
    if not urls_candidatas:
        raise HTTPException(
            status_code=404,
            detail=(
                "Nenhuma imagem encontrada nesta página. O site provavelmente carrega imagens via JavaScript "
                "e elas não estão no HTML estático. Tente copiar a URL direta de uma imagem ou enviar um print."
            ),
        )

    resultados = []
    erros = []
    videos_detectados = []
    prompt_final = _prompt_final(usuario, db)

    for url_img in urls_candidatas:
        if len(resultados) >= body.limite:
            break

        if _parece_url_video(url_img):
            videos_detectados.append(url_img)
            continue

        try:
            dados, mime_type, url_final = await baixar_imagem_validada(url_img, referer=url_base)
            data_uri = imagem_para_data_uri(dados, mime_type)
            descricao = chamar_modelo(data_uri, prompt_final)
        except HTTPException as e:
            erros.append({"url": url_img, "erro": e.detail[:250]})
            continue
        except Exception as e:
            erros.append({"url": url_img, "erro": str(e)[:250]})
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
        resultados.append({"id": registro.id, "url_imagem": url_final, "descricao": descricao})

    if not resultados:
        detalhe = (
            f"Encontrei {len(urls_candidatas)} URL(s) de imagem, mas não consegui baixar nenhuma válida. "
            "Possíveis causas: o site bloqueia download automático, usa proteção de player, exige login/captcha, "
            "ou as imagens são carregadas por script. "
            "Tente enviar uma imagem direta, print da tela ou arquivo."
        )
        if erros:
            detalhe += f" Primeiro erro: {erros[0]['erro']}"
        raise HTTPException(status_code=502, detail=detalhe)

    return {
        "pagina_url": body.url,
        "total_imagens_encontradas": len(urls_candidatas),
        "total_descritas": len(resultados),
        "videos_detectados": videos_detectados[:3],
        "erros": erros[:5],
        "imagens": resultados,
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
