import os

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

import models
import schemas
from auth import criar_token, hash_senha, verificar_senha
from database import Base, engine, get_db
from routers import documentos, estabelecimentos, feedback, imagens, internet, pessoas, skills, usuario, videos

Base.metadata.create_all(bind=engine)

GROK_API_KEY = os.environ.get("GROK_API_KEY")
if not GROK_API_KEY:
    print("[Descritoria] AVISO: GROK_API_KEY não definida. Descrições de imagem não funcionarão.")

app = FastAPI(
    title="Descritoria API",
    description=(
        "Backend do Descritoria — aplicativo de acessibilidade visual para pessoas cegas. "
        "Oferece descrição de imagens, vídeos e documentos com IA, loja de skills, "
        "diretório de estabelecimentos acessíveis e sistema de contas com PerceptMoney."
    ),
    version="1.0.0",
    contact={"name": "Descritoria", "email": "contato@descritoria.app"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Frontend acessível ────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
def pagina_principal():
    return FileResponse("static/index.html")


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(imagens.router)
app.include_router(internet.router)
app.include_router(videos.router)
app.include_router(documentos.router)
app.include_router(estabelecimentos.router)
app.include_router(skills.router)
app.include_router(usuario.router)
app.include_router(feedback.router)
app.include_router(pessoas.router)


# ── Diagnóstico de modelo ─────────────────────────────────────────────────────

@app.get("/modelo", response_class=HTMLResponse, include_in_schema=False)
def modelo_ativo():
    modelo = imagens.MODELO_IMAGEM
    return HTMLResponse(f"""<!DOCTYPE html>
<html lang="pt-BR">
<head><meta charset="UTF-8"><title>Status do Modelo</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body style="font-family:sans-serif;padding:2rem;background:#111;color:#fff;">
<h1 style="color:green;">Grok Vision ativo</h1>
<p style="font-size:1.2rem;">Modelo: {modelo} — sem censura, alta qualidade.</p>
<p><a href="/" style="color:#4af;">Voltar ao app</a></p>
</body></html>""")


# ── Autenticação ──────────────────────────────────────────────────────────────

@app.post(
    "/auth/registro",
    summary="Criar nova conta",
    status_code=status.HTTP_201_CREATED,
    tags=["Autenticação"],
)
def registro(body: schemas.RegistroRequest, db: Session = Depends(get_db)):
    existente = db.query(models.Usuario).filter(models.Usuario.email == body.email).first()
    if existente:
        raise HTTPException(status_code=409, detail="E-mail já cadastrado.")
    usuario = models.Usuario(
        nome=body.nome,
        email=body.email,
        senha_hash=hash_senha(body.senha),
    )
    db.add(usuario)
    db.commit()
    db.refresh(usuario)
    token = criar_token({"sub": str(usuario.id)})
    return {"access_token": token, "token_type": "bearer", "usuario_id": usuario.id}


@app.post("/auth/login", summary="Login com e-mail e senha", tags=["Autenticação"])
def login(body: schemas.LoginRequest, db: Session = Depends(get_db)):
    usuario = db.query(models.Usuario).filter(models.Usuario.email == body.email).first()
    if not usuario or not verificar_senha(body.senha, usuario.senha_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="E-mail ou senha incorretos.",
        )
    token = criar_token({"sub": str(usuario.id)})
    return {"access_token": token, "token_type": "bearer", "usuario_id": usuario.id}


@app.post("/auth/token/refresh", summary="Renovar token de acesso", tags=["Autenticação"])
def renovar_token(body: schemas.TokenRefreshRequest, db: Session = Depends(get_db)):
    from auth import decodificar_token
    payload = decodificar_token(body.token)
    usuario_id = payload.get("sub")
    if not usuario_id:
        raise HTTPException(status_code=401, detail="Token inválido.")
    usuario = db.query(models.Usuario).filter(models.Usuario.id == int(usuario_id)).first()
    if not usuario:
        raise HTTPException(status_code=401, detail="Usuário não encontrado.")
    novo_token = criar_token({"sub": str(usuario.id)})
    return {"access_token": novo_token, "token_type": "bearer"}


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/saude", summary="Verificação de saúde da API", tags=["Sistema"])
def saude():
    return {"status": "ok", "api": "Descritoria", "versao": "1.0.0"}


# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
