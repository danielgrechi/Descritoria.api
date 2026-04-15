import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import models
import schemas
from auth import obter_usuario_atual
from database import get_db

router = APIRouter(prefix="/skills", tags=["Skills"])

CATEGORIAS_SKILLS = [
    "Acessibilidade", "Entretenimento", "Educação", "Saúde",
    "Trabalho", "Culinária", "Esportes", "Notícias", "Outros",
]


@router.get("", summary="Catálogo de skills disponíveis")
def listar_skills(
    categoria: Optional[str] = None,
    busca: Optional[str] = None,
    ordem: Optional[str] = None,  # "em_alta", "avaliacao", "promocao"
    limite: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    query = db.query(models.Skill).filter(models.Skill.publicado == True)
    if categoria:
        query = query.filter(models.Skill.categoria == categoria)
    if busca:
        termo = f"%{busca}%"
        query = query.filter(
            models.Skill.nome.ilike(termo) | models.Skill.descricao.ilike(termo)
        )
    if ordem == "em_alta":
        query = query.filter(models.Skill.em_destaque == True)
    elif ordem == "avaliacao":
        query = query.order_by(models.Skill.avaliacao_media.desc())
    elif ordem == "promocao":
        query = query.filter(models.Skill.em_promocao == True)
    else:
        query = query.order_by(models.Skill.total_assinantes.desc())

    total = query.count()
    items = query.offset(offset).limit(limite).all()
    return {"total": total, "items": [schemas.SkillResponse.from_orm(s) for s in items]}


@router.get("/categorias", summary="Categorias de skills")
def listar_categorias():
    return {"categorias": CATEGORIAS_SKILLS}


@router.get("/minhas", summary="Skills do usuário (criadas e assinadas)")
def minhas_skills(
    filtro: Optional[str] = None,  # "criadas" ou "assinadas"
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    criadas, assinadas = [], []

    if filtro in (None, "criadas"):
        criadas = db.query(models.Skill).filter(
            models.Skill.criador_id == usuario.id
        ).all()

    if filtro in (None, "assinadas"):
        assinaturas = db.query(models.SkillAssinatura).filter(
            models.SkillAssinatura.usuario_id == usuario.id
        ).all()
        ids_assinadas = [a.skill_id for a in assinaturas]
        assinadas = db.query(models.Skill).filter(models.Skill.id.in_(ids_assinadas)).all()

    return {
        "criadas": [schemas.SkillResponse.from_orm(s) for s in criadas],
        "assinadas": [schemas.SkillResponse.from_orm(s) for s in assinadas],
    }


@router.get("/{skill_id}", summary="Detalhes de uma skill")
def obter_skill(skill_id: int, db: Session = Depends(get_db)):
    skill = db.query(models.Skill).filter(models.Skill.id == skill_id).first()
    if not skill:
        raise HTTPException(status_code=404, detail="Skill não encontrada.")
    return schemas.SkillResponse.from_orm(skill)


@router.post("", summary="Publicar nova skill (construtor)", status_code=201)
def criar_skill(
    body: schemas.SkillCreate,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    skill = models.Skill(
        criador_id=usuario.id,
        **body.model_dump(),
    )
    db.add(skill)
    db.commit()
    db.refresh(skill)
    return schemas.SkillResponse.from_orm(skill)


@router.post("/{skill_id}/assinar", summary="Assinar/comprar uma skill")
def assinar_skill(
    skill_id: int,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    skill = db.query(models.Skill).filter(
        models.Skill.id == skill_id, models.Skill.publicado == True
    ).first()
    if not skill:
        raise HTTPException(status_code=404, detail="Skill não encontrada.")

    ja_assinou = db.query(models.SkillAssinatura).filter(
        models.SkillAssinatura.usuario_id == usuario.id,
        models.SkillAssinatura.skill_id == skill_id,
    ).first()
    if ja_assinou:
        raise HTTPException(status_code=409, detail="Você já assinou esta skill.")

    if skill.preco > 0:
        if usuario.saldo_perceptmoney < skill.preco:
            raise HTTPException(
                status_code=402,
                detail=f"Saldo insuficiente. Necessário: {skill.preco} PerceptMoney.",
            )
        usuario.saldo_perceptmoney -= skill.preco
        criador = db.query(models.Usuario).filter(models.Usuario.id == skill.criador_id).first()
        if criador:
            criador.saldo_perceptmoney += skill.preco

        db.add(models.Transacao(
            usuario_id=usuario.id,
            tipo=models.TipoTransacao.debito,
            valor=skill.preco,
            descricao=f"Assinatura da skill: {skill.nome}",
            referencia_id=skill_id,
        ))

    assinatura = models.SkillAssinatura(usuario_id=usuario.id, skill_id=skill_id)
    skill.total_assinantes += 1
    db.add(assinatura)
    db.commit()

    return {"mensagem": f"Skill '{skill.nome}' assinada com sucesso."}


@router.delete("/{skill_id}", summary="Remover skill própria")
def deletar_skill(
    skill_id: int,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    skill = db.query(models.Skill).filter(
        models.Skill.id == skill_id, models.Skill.criador_id == usuario.id
    ).first()
    if not skill:
        raise HTTPException(status_code=404, detail="Skill não encontrada ou sem permissão.")
    db.delete(skill)
    db.commit()
    return {"mensagem": "Skill removida com sucesso."}
