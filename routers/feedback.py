from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import models
import schemas
from auth import obter_usuario_atual
from database import get_db

router = APIRouter(prefix="/feedback", tags=["Feedback"])


def _verificar_descricao(descricao_id: int, usuario_id: int, db: Session) -> models.Descricao:
    registro = db.query(models.Descricao).filter(
        models.Descricao.id == descricao_id,
        models.Descricao.usuario_id == usuario_id,
    ).first()
    if not registro:
        raise HTTPException(status_code=404, detail="Descrição não encontrada.")
    return registro


@router.post("/bom", summary="Registrar feedback positivo")
def feedback_bom(
    body: schemas.FeedbackBomRequest,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    _verificar_descricao(body.descricao_id, usuario.id, db)

    ja_existe = db.query(models.Feedback).filter(
        models.Feedback.descricao_id == body.descricao_id,
        models.Feedback.usuario_id == usuario.id,
    ).first()
    if ja_existe:
        raise HTTPException(status_code=409, detail="Feedback já registrado para esta descrição.")

    feedback = models.Feedback(
        descricao_id=body.descricao_id,
        usuario_id=usuario.id,
        tipo=models.TipoFeedback.bom,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return schemas.FeedbackResponse.from_orm(feedback)


@router.post("/ruim", summary="Registrar feedback negativo com descrição do erro")
def feedback_ruim(
    body: schemas.FeedbackRuimRequest,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    _verificar_descricao(body.descricao_id, usuario.id, db)

    ja_existe = db.query(models.Feedback).filter(
        models.Feedback.descricao_id == body.descricao_id,
        models.Feedback.usuario_id == usuario.id,
    ).first()
    if ja_existe:
        raise HTTPException(status_code=409, detail="Feedback já registrado para esta descrição.")

    feedback = models.Feedback(
        descricao_id=body.descricao_id,
        usuario_id=usuario.id,
        tipo=models.TipoFeedback.ruim,
        comentario=body.comentario,
    )
    db.add(feedback)

    # Aprende com o feedback: adiciona a preferência ao estilo do usuário
    if body.comentario:
        estilo_atual = usuario.estilo_descricao or ""
        nova_instrucao = body.comentario.strip()
        if nova_instrucao not in estilo_atual:
            if estilo_atual:
                usuario.estilo_descricao = estilo_atual + "\n- " + nova_instrucao
            else:
                usuario.estilo_descricao = "- " + nova_instrucao

    db.commit()
    db.refresh(feedback)
    return {
        **schemas.FeedbackResponse.from_orm(feedback).dict(),
        "aprendizado": "Sua preferência foi salva e será usada nas próximas descrições.",
    }
