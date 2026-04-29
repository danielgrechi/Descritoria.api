from __future__ import annotations
import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import models
import schemas
from auth import obter_usuario_atual
from database import get_db

router = APIRouter(prefix="/feedback", tags=["Feedback"])


def _verificar_descricao(descricao_id: int, usuario_id: int, db: Session) -> models.Descricao:
    registro = (
        db.query(models.Descricao)
        .filter(models.Descricao.id == descricao_id, models.Descricao.usuario_id == usuario_id)
        .first()
    )
    if not registro:
        raise HTTPException(status_code=404, detail="Descrição não encontrada.")
    return registro


def _preferencias_dict(usuario: models.Usuario) -> dict:
    try:
        return json.loads(usuario.preferencias_extraidas or "{}")
    except Exception:
        return {}


def _salvar_preferencias(usuario: models.Usuario, preferencias: dict) -> None:
    usuario.preferencias_extraidas = json.dumps(preferencias, ensure_ascii=False)


def _atualizar_aprendizado_usuario(usuario: models.Usuario, tipo: models.TipoFeedback, comentario: str | None) -> None:
    preferencias = _preferencias_dict(usuario)
    preferencias.setdefault("elogios", [])
    preferencias.setdefault("correcoes", [])
    preferencias.setdefault("ultima_sintese", "")

    comentario_limpo = (comentario or "").strip()
    if comentario_limpo:
        destino = "elogios" if tipo == models.TipoFeedback.bom else "correcoes"
        if comentario_limpo not in preferencias[destino]:
            preferencias[destino].append(comentario_limpo[:500])
            preferencias[destino] = preferencias[destino][-20:]

    correcoes = preferencias.get("correcoes", [])
    elogios = preferencias.get("elogios", [])

    linhas = [
        "Perfil de descrição aprendido com o uso:",
        "- Priorizar descrições úteis para uma pessoa cega, com linguagem natural e direta.",
        "- Manter regra rígida de zero alucinação: não inventar elementos, pessoas, emoções, contexto ou textos não legíveis.",
        "- Quando houver dúvida, dizer explicitamente que não é possível confirmar pela imagem.",
    ]

    if elogios:
        linhas.append("- O usuário valorizou estes pontos em descrições anteriores:")
        linhas.extend(f"  - {item}" for item in elogios[-8:])

    if correcoes:
        linhas.append("- O usuário pediu correções ou ajustes nestes pontos:")
        linhas.extend(f"  - {item}" for item in correcoes[-12:])

    usuario.perfil_aprendizado_ia = "\n".join(linhas)[:4000]
    preferencias["ultima_sintese"] = usuario.perfil_aprendizado_ia
    usuario.ultima_atualizacao_perfil = datetime.utcnow()
    _salvar_preferencias(usuario, preferencias)


def _registrar_feedback_unico(
    descricao_id: int,
    usuario: models.Usuario,
    db: Session,
    tipo: models.TipoFeedback,
    comentario: str | None = None,
):
    _verificar_descricao(descricao_id, usuario.id, db)

    ja_existe = (
        db.query(models.Feedback)
        .filter(models.Feedback.descricao_id == descricao_id, models.Feedback.usuario_id == usuario.id)
        .first()
    )

    if ja_existe:
        ja_existe.tipo = tipo
        ja_existe.comentario = comentario
        feedback = ja_existe
    else:
        feedback = models.Feedback(
            descricao_id=descricao_id,
            usuario_id=usuario.id,
            tipo=tipo,
            comentario=comentario,
        )
        db.add(feedback)

    _atualizar_aprendizado_usuario(usuario, tipo, comentario)

    db.commit()
    db.refresh(feedback)
    db.refresh(usuario)

    return feedback


@router.post("/bom", summary="Registrar feedback positivo")
def feedback_bom(
    body: schemas.FeedbackBomRequest,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    feedback = _registrar_feedback_unico(
        descricao_id=body.descricao_id,
        usuario=usuario,
        db=db,
        tipo=models.TipoFeedback.bom,
        comentario=body.comentario,
    )
    return {
        **schemas.FeedbackResponse.from_orm(feedback).model_dump(),
        "aprendizado": "Obrigado. Esse retorno foi incorporado ao seu perfil de descrição.",
        "perfil_aprendizado_ia": usuario.perfil_aprendizado_ia or "",
    }


@router.post("/ruim", summary="Registrar feedback negativo com descrição do erro")
def feedback_ruim(
    body: schemas.FeedbackRuimRequest,
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    feedback = _registrar_feedback_unico(
        descricao_id=body.descricao_id,
        usuario=usuario,
        db=db,
        tipo=models.TipoFeedback.ruim,
        comentario=body.comentario,
    )
    return {
        **schemas.FeedbackResponse.from_orm(feedback).model_dump(),
        "aprendizado": "Sua correção foi salva e será usada para ajustar as próximas descrições.",
        "perfil_aprendizado_ia": usuario.perfil_aprendizado_ia or "",
    }
