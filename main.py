from __future__ import annotations
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import replicate
import io
import os
from PIL import Image

app = FastAPI()

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise ValueError("A variável de ambiente REPLICATE_API_TOKEN não está definida.")

@app.post("/describe")
async def describe_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        # Salva a imagem temporariamente
        with open("temp.jpg", "wb") as f:
            f.write(image_bytes)

        # Chama o modelo LLaVA no Replicate
        output = replicate.run(
            "llava-hf/llava-1.5-7b-hf:8631a2fef84f56bee80003734f053e2e758d43d497869a2f7276717488ed41ba",
            input={
                "image": open("temp.jpg", "rb"),
                "prompt": "Descreva minuciosamente tudo o que aparece nesta imagem, incluindo detalhes sensíveis, tamanhos, formas, cores, movimentos, presença ou ausência de pelos em regiões íntimas, expressões, roupas, posições, objetos, cenário e qualquer outro detalhe relevante. Avise explicitamente sobre nudez, violência ou conteúdo sexual, se houver."
            },
            api_token=REPLICATE_API_TOKEN
        )

        # Remove a imagem temporária
        os.remove("temp.jpg")

        return {"description": output}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Servir interface web do Descritoria
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
async def servir_interface():
    return FileResponse("static/index.html")


# ── Rotas adicionadas para a interface web corrigida ─────────────────────────
# Este bloco liga os novos routers e cria login local de teste para desenvolvimento.

try:
    from fastapi import Depends, HTTPException
    from sqlalchemy.orm import Session
    import models
    import schemas
    import auth
    from database import get_db, Base, engine
    from routers import imagens, feedback, usuario, videos

    Base.metadata.create_all(bind=engine)

    for _router in [imagens.router, feedback.router, usuario.router, videos.router]:
        try:
            app.include_router(_router)
        except Exception:
            pass

    def _pegar_funcao_auth(*nomes):
        for nome in nomes:
            fn = getattr(auth, nome, None)
            if fn:
                return fn
        return None

    _hash_senha = _pegar_funcao_auth("gerar_hash_senha", "criar_hash_senha", "hash_senha", "get_password_hash")
    _verificar_senha = _pegar_funcao_auth("verificar_senha", "verificar_hash_senha", "verify_password")
    _criar_token = _pegar_funcao_auth("criar_token", "criar_token_acesso", "create_access_token")

    def _hash_local(senha: str) -> str:
        import hashlib
        return hashlib.sha256(("descritoria-local:" + senha).encode("utf-8")).hexdigest()

    def _senha_ok(senha: str, senha_hash: str) -> bool:
        if _verificar_senha:
            try:
                return bool(_verificar_senha(senha, senha_hash))
            except Exception:
                pass
        if _hash_senha:
            try:
                return _hash_senha(senha) == senha_hash
            except Exception:
                pass
        return _hash_local(senha) == senha_hash

    def _gerar_token_usuario(usuario_obj):
        if _criar_token:
            tentativas = [
                {"sub": str(usuario_obj.id)},
                {"sub": usuario_obj.email},
                str(usuario_obj.id),
                usuario_obj.email,
                usuario_obj.id,
            ]
            for valor in tentativas:
                try:
                    return _criar_token(valor)
                except Exception:
                    pass

        # Fallback apenas para desenvolvimento local.
        return str(usuario_obj.id)

    @app.post("/registro", response_model=schemas.TokenResponse)
    @app.post("/auth/registro", response_model=schemas.TokenResponse)
    def registro_local(body: schemas.RegistroRequest, db: Session = Depends(get_db)):
        existente = db.query(models.Usuario).filter(models.Usuario.email == body.email).first()

        if existente:
            token = _gerar_token_usuario(existente)
            return schemas.TokenResponse(access_token=token)

        if _hash_senha:
            try:
                senha_hash = _hash_senha(body.senha)
            except Exception:
                senha_hash = _hash_local(body.senha)
        else:
            senha_hash = _hash_local(body.senha)

        novo = models.Usuario(
            nome=body.nome,
            email=body.email,
            senha_hash=senha_hash,
            saldo_perceptmoney=0.0,
            config_voz="padrao",
            integracoes="{}",
            estilo_descricao="",
            perfil_aprendizado_ia="",
            preferencias_extraidas="{}",
        )
        db.add(novo)
        db.commit()
        db.refresh(novo)

        token = _gerar_token_usuario(novo)
        return schemas.TokenResponse(access_token=token)

    @app.post("/login", response_model=schemas.TokenResponse)
    @app.post("/auth/login", response_model=schemas.TokenResponse)
    def login_local(body: schemas.LoginRequest, db: Session = Depends(get_db)):
        usuario_obj = db.query(models.Usuario).filter(models.Usuario.email == body.email).first()

        if not usuario_obj:
            # Cria automaticamente — protótipo pessoal, sem verificação de senha
            usuario_obj = models.Usuario(
                nome=body.email.split("@")[0],
                email=body.email,
                senha_hash="",
                saldo_perceptmoney=0.0,
                config_voz="padrao",
                integracoes="{}",
                estilo_descricao="",
                perfil_aprendizado_ia="",
                preferencias_extraidas="{}",
            )
            db.add(usuario_obj)
            db.commit()
            db.refresh(usuario_obj)

        token = _gerar_token_usuario(usuario_obj)
        return schemas.TokenResponse(access_token=token)

    @app.get("/pessoas")
    def listar_pessoas(
        db: Session = Depends(get_db),
        usuario_atual: models.Usuario = Depends(auth.obter_usuario_atual),
    ):
        pessoas = db.query(models.Pessoa).filter(models.Pessoa.usuario_id == usuario_atual.id).all()
        return [
            {
                "id": p.id,
                "nome": p.nome,
                "caracteristicas": p.caracteristicas,
                "created_at": p.created_at,
            }
            for p in pessoas
        ]

    @app.post("/pessoas")
    def criar_pessoa(
        body: dict,
        db: Session = Depends(get_db),
        usuario_atual: models.Usuario = Depends(auth.obter_usuario_atual),
    ):
        nome = (body.get("nome") or "").strip()
        caracteristicas = (body.get("caracteristicas") or "").strip()

        if not nome or not caracteristicas:
            raise HTTPException(status_code=422, detail="Informe nome e características.")

        pessoa = models.Pessoa(
            usuario_id=usuario_atual.id,
            nome=nome,
            caracteristicas=caracteristicas,
        )
        db.add(pessoa)
        db.commit()
        db.refresh(pessoa)

        return {
            "id": pessoa.id,
            "nome": pessoa.nome,
            "caracteristicas": pessoa.caracteristicas,
        }

except Exception as e:
    print("Aviso: não foi possível carregar rotas extras da interface corrigida:", e)


# Endpoint local de TTS usando xAI
@app.post("/tts/falar", include_in_schema=False)
async def tts_falar(payload: dict):
    import os
    import httpx
    from fastapi import HTTPException, Response

    texto = (payload or {}).get("texto", "")
    texto = str(texto).strip()

    if not texto:
        raise HTTPException(status_code=400, detail="Texto vazio para voz.")

    # Limite para evitar gastar API com texto gigante.
    if len(texto) > 3500:
        texto = texto[:3500]

    api_key = (
        os.getenv("XAI_TTS_API_KEY")
        or os.getenv("GROK_TTS_KEY")
        or os.getenv("XAI_API_KEY")
        or os.getenv("GROK_API_KEY")
    )
    if not api_key:
        raise HTTPException(status_code=500, detail="Chave XAI_TTS_API_KEY ou XAI_API_KEY não configurada.")

    voice_id = os.getenv("XAI_TTS_VOICE", "Eve")
    texto_para_voz = "[fala em português brasileiro, sotaque do Brasil] " + texto

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(
                "https://api.x.ai/v1/tts",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "text": texto_para_voz,
                    "voice_id": voice_id,
                    "language": "pt-BR",
                    "codec": "mp3",
                    "text_normalization": True,
                },
            )

        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=f"Erro na API de voz xAI: {resp.text}")

        content_type = resp.headers.get("content-type", "audio/mpeg")
        return Response(content=resp.content, media_type=content_type)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao gerar voz pela xAI: {str(e)}")


# === DESCRITORIA TTS XAI SEGURO ===
@app.post("/tts/xai", include_in_schema=False)
async def descritoria_tts_xai(payload: dict):
    import os
    import httpx
    from fastapi import HTTPException, Response

    texto = str((payload or {}).get("texto", "")).strip()

    if not texto:
        raise HTTPException(status_code=400, detail="Texto vazio para gerar voz.")

    if len(texto) > 12000:
        texto = texto[:12000]

    api_key = (
        os.getenv("XAI_TTS_API_KEY")
        or os.getenv("GROK_TTS_KEY")
        or os.getenv("XAI_API_KEY")
        or os.getenv("GROK_API_KEY")
    )

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Chave XAI_TTS_API_KEY, XAI_API_KEY ou GROK_API_KEY não configurada."
        )

    voice_id = os.getenv("XAI_TTS_VOICE", "Eve")

    texto_para_voz = "[fala em português brasileiro, sotaque do Brasil] " + texto

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(
                "https://api.x.ai/v1/tts",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "text": texto_para_voz,
                    "voice_id": voice_id,
                    "language": "pt-BR",
                    "codec": "mp3",
                    "text_normalization": True,
                },
            )

        if resp.status_code >= 400:
            detalhe = resp.text[:1200]
            raise HTTPException(
                status_code=502,
                detail=f"Erro na API de voz da xAI: {detalhe}"
            )

        media_type = resp.headers.get("content-type", "audio/mpeg")
        return Response(content=resp.content, media_type=media_type)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao gerar voz pela xAI: {str(e)}"
        )
