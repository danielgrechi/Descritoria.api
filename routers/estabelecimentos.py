from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import models
import schemas
from auth import obter_usuario_atual
from database import get_db

router = APIRouter(prefix="/estabelecimentos", tags=["Estabelecimentos"])

CATEGORIAS_DISPONIVEIS = [
    "Restaurante", "Farmácia", "Banco", "Hospital", "Clínica",
    "Supermercado", "Shopping", "Hotel", "Academia", "Transporte",
    "Educação", "Cultura", "Lazer", "Serviços", "Outros",
]


@router.get("/categorias", summary="Lista categorias disponíveis")
def listar_categorias():
    return {"categorias": CATEGORIAS_DISPONIVEIS}


@router.get("", summary="Lista estabelecimentos amigos de PcD")
def listar_estabelecimentos(
    categoria: Optional[str] = None,
    busca: Optional[str] = None,
    limite: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    query = db.query(models.Estabelecimento)
    if categoria:
        query = query.filter(models.Estabelecimento.categoria == categoria)
    if busca:
        termo = f"%{busca}%"
        query = query.filter(
            models.Estabelecimento.nome.ilike(termo)
            | models.Estabelecimento.descricao.ilike(termo)
            | models.Estabelecimento.endereco.ilike(termo)
        )
    total = query.count()
    items = (
        query.order_by(models.Estabelecimento.avaliacao_media.desc())
        .offset(offset)
        .limit(limite)
        .all()
    )
    return {
        "total": total,
        "items": [schemas.EstabelecimentoResponse.from_orm(e) for e in items],
    }


@router.get("/{estabelecimento_id}", summary="Detalhes de um estabelecimento")
def obter_estabelecimento(
    estabelecimento_id: int,
    db: Session = Depends(get_db),
):
    estab = db.query(models.Estabelecimento).filter(
        models.Estabelecimento.id == estabelecimento_id
    ).first()
    if not estab:
        raise HTTPException(status_code=404, detail="Estabelecimento não encontrado.")
    return schemas.EstabelecimentoResponse.from_orm(estab)


@router.post("", summary="Cadastrar novo estabelecimento", status_code=201)
def cadastrar_estabelecimento(
    body: schemas.EstabelecimentoCreate,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    estab = models.Estabelecimento(**body.model_dump())
    db.add(estab)
    db.commit()
    db.refresh(estab)
    return schemas.EstabelecimentoResponse.from_orm(estab)


@router.put("/{estabelecimento_id}", summary="Atualizar estabelecimento")
def atualizar_estabelecimento(
    estabelecimento_id: int,
    body: schemas.EstabelecimentoUpdate,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    estab = db.query(models.Estabelecimento).filter(
        models.Estabelecimento.id == estabelecimento_id
    ).first()
    if not estab:
        raise HTTPException(status_code=404, detail="Estabelecimento não encontrado.")
    for campo, valor in body.model_dump(exclude_none=True).items():
        setattr(estab, campo, valor)
    db.commit()
    db.refresh(estab)
    return schemas.EstabelecimentoResponse.from_orm(estab)


@router.post("/{estabelecimento_id}/avaliar", summary="Avaliar acessibilidade do estabelecimento")
def avaliar_estabelecimento(
    estabelecimento_id: int,
    body: schemas.AvaliacaoRequest,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    estab = db.query(models.Estabelecimento).filter(
        models.Estabelecimento.id == estabelecimento_id
    ).first()
    if not estab:
        raise HTTPException(status_code=404, detail="Estabelecimento não encontrado.")

    total = estab.total_avaliacoes
    media_atual = estab.avaliacao_media
    nova_media = ((media_atual * total) + body.nota) / (total + 1)

    estab.avaliacao_media = round(nova_media, 2)
    estab.total_avaliacoes = total + 1
    db.commit()
    db.refresh(estab)

    return {
        "mensagem": "Avaliação registrada com sucesso.",
        "avaliacao_media": estab.avaliacao_media,
        "total_avaliacoes": estab.total_avaliacoes,
    }
