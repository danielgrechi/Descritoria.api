import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import models
import schemas
from auth import hash_senha, obter_usuario_atual, criar_token, verificar_senha
from database import get_db

router = APIRouter(prefix="/usuario", tags=["Usuário"])

REDES_SOCIAIS_SUPORTADAS = ["instagram", "twitter", "facebook", "tiktok", "whatsapp"]


@router.get("/perfil", summary="Obter perfil do usuário")
def obter_perfil(usuario: models.Usuario = Depends(obter_usuario_atual)):
    return schemas.PerfilResponse.from_orm(usuario)


@router.put("/perfil", summary="Atualizar perfil")
def atualizar_perfil(
    body: schemas.PerfilUpdateRequest,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    if body.nome:
        usuario.nome = body.nome
    if body.config_voz:
        usuario.config_voz = body.config_voz
    db.commit()
    db.refresh(usuario)
    return schemas.PerfilResponse.from_orm(usuario)


@router.get("/saldo", summary="Saldo em PerceptMoney")
def obter_saldo(usuario: models.Usuario = Depends(obter_usuario_atual)):
    return schemas.SaldoResponse(saldo=usuario.saldo_perceptmoney)


@router.get("/extrato", summary="Histórico de transações")
def extrato(
    limite: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    query = db.query(models.Transacao).filter(models.Transacao.usuario_id == usuario.id)
    total = query.count()
    items = query.order_by(models.Transacao.created_at.desc()).offset(offset).limit(limite).all()
    return {
        "total": total,
        "items": [schemas.TransacaoResponse.from_orm(t) for t in items],
    }


@router.post("/cartao", summary="Cadastrar cartão de crédito", status_code=201)
def cadastrar_cartao(
    body: schemas.CartaoRequest,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    # Detecta bandeira pelos primeiros dígitos
    numero = body.numero
    if numero.startswith("4"):
        bandeira = "Visa"
    elif numero.startswith(("51", "52", "53", "54", "55")):
        bandeira = "Mastercard"
    elif numero.startswith(("34", "37")):
        bandeira = "Amex"
    else:
        bandeira = "Outra"

    # Em produção, tokenizar o cartão via gateway (ex: Stripe, PagSeguro)
    token_gateway = f"tok_{numero[-4:]}_{usuario.id}"

    # Define como principal se for o primeiro
    eh_primeiro = not db.query(models.Cartao).filter(
        models.Cartao.usuario_id == usuario.id
    ).first()

    cartao = models.Cartao(
        usuario_id=usuario.id,
        ultimos4=numero[-4:],
        bandeira=bandeira,
        token_gateway=token_gateway,
        principal=eh_primeiro,
    )
    db.add(cartao)
    db.commit()
    db.refresh(cartao)
    return schemas.CartaoResponse.from_orm(cartao)


@router.get("/cartoes", summary="Listar cartões cadastrados")
def listar_cartoes(
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    cartoes = db.query(models.Cartao).filter(models.Cartao.usuario_id == usuario.id).all()
    return [schemas.CartaoResponse.from_orm(c) for c in cartoes]


@router.get("/configuracoes", summary="Obter configurações do usuário")
def obter_configuracoes(usuario: models.Usuario = Depends(obter_usuario_atual)):
    try:
        integracoes = json.loads(usuario.integracoes or "{}")
    except Exception:
        integracoes = {}
    return schemas.ConfiguracoesResponse(
        config_voz=usuario.config_voz,
        integracoes=integracoes,
    )


@router.put("/configuracoes", summary="Atualizar configurações")
def atualizar_configuracoes(
    body: schemas.ConfiguracoesUpdateRequest,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    if body.config_voz:
        usuario.config_voz = body.config_voz
    db.commit()
    try:
        integracoes = json.loads(usuario.integracoes or "{}")
    except Exception:
        integracoes = {}
    return schemas.ConfiguracoesResponse(
        config_voz=usuario.config_voz,
        integracoes=integracoes,
    )


@router.post("/integracoes/{rede}", summary="Vincular rede social")
def vincular_rede_social(
    rede: str,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    if rede.lower() not in REDES_SOCIAIS_SUPORTADAS:
        raise HTTPException(
            status_code=422,
            detail=f"Rede não suportada. Opções: {', '.join(REDES_SOCIAIS_SUPORTADAS)}",
        )
    try:
        integracoes = json.loads(usuario.integracoes or "{}")
    except Exception:
        integracoes = {}

    # Em produção, iniciar fluxo OAuth da rede social
    integracoes[rede.lower()] = {"vinculado": True, "status": "pendente_oauth"}
    usuario.integracoes = json.dumps(integracoes)
    db.commit()

    return {
        "mensagem": f"Integração com {rede} iniciada. Complete o processo OAuth no app.",
        "rede": rede.lower(),
        "status": "pendente_oauth",
    }
