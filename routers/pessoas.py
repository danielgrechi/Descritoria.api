from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

import models
from auth import obter_usuario_atual
from database import get_db

router = APIRouter(prefix="/pessoas", tags=["Memória de Pessoas"])


class PessoaCreate(BaseModel):
    nome: str
    caracteristicas: str


class PessoaResponse(BaseModel):
    id: int
    nome: str
    caracteristicas: str
    created_at: datetime

    class Config:
        from_attributes = True


class PessoaUpdate(BaseModel):
    nome: Optional[str] = None
    caracteristicas: Optional[str] = None


@router.post("", summary="Salvar pessoa conhecida na memória", status_code=201)
def salvar_pessoa(
    body: PessoaCreate,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    existente = db.query(models.Pessoa).filter(
        models.Pessoa.usuario_id == usuario.id,
        models.Pessoa.nome == body.nome,
    ).first()

    if existente:
        existente.caracteristicas = body.caracteristicas
        existente.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existente)
        return PessoaResponse.from_orm(existente)

    pessoa = models.Pessoa(
        usuario_id=usuario.id,
        nome=body.nome,
        caracteristicas=body.caracteristicas,
    )
    db.add(pessoa)
    db.commit()
    db.refresh(pessoa)
    return PessoaResponse.from_orm(pessoa)


@router.get("", summary="Listar pessoas conhecidas", response_model=List[PessoaResponse])
def listar_pessoas(
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    return db.query(models.Pessoa).filter(
        models.Pessoa.usuario_id == usuario.id
    ).order_by(models.Pessoa.nome).all()


@router.delete("/{pessoa_id}", summary="Remover pessoa da memória")
def remover_pessoa(
    pessoa_id: int,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    pessoa = db.query(models.Pessoa).filter(
        models.Pessoa.id == pessoa_id,
        models.Pessoa.usuario_id == usuario.id,
    ).first()
    if not pessoa:
        raise HTTPException(status_code=404, detail="Pessoa não encontrada.")
    db.delete(pessoa)
    db.commit()
    return {"ok": True}
