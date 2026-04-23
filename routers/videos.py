import base64
import hashlib
import os

import replicate
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

import models
from auth import decodificar_token, obter_usuario_atual
from database import get_db

router = APIRouter(prefix="/videos", tags=["Vídeos"])

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")
MODELO_IMAGEM = "yorickvp/llava-13b:80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb"

PROMPT_VIDEO = (
    "Você é um assistente de acessibilidade visual para pessoas cegas. "
    "Este frame é de um vídeo. Descreva em português do Brasil o que está acontecendo: "
    "pessoas, ações, objetos, cenário, movimento e qualquer detalhe relevante. "
    "Se houver nudez, violência ou conteúdo sexual, informe no início com precisão clínica. "
    "Seja objetivo e use parágrafos fluentes."
)

PROMPT_AOVIVO = (
    "Você é um assistente de acessibilidade em tempo real para uma pessoa cega. "
    "Descreva brevemente (2-3 frases) o que está na cena atual capturada pela câmera. "
    "Foque no mais importante: pessoas, objetos próximos, texto visível, obstáculos. "
    "Responda em português do Brasil."
)


def frame_para_data_uri(dados: bytes) -> str:
    b64 = base64.b64encode(dados).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


@router.post("/descrever", summary="Descreve um vídeo frame a frame")
async def descrever_video(
    arquivo: UploadFile = File(...),
    db: Session = Depends(get_db),
    usuario: models.Usuario = Depends(obter_usuario_atual),
):
    dados = await arquivo.read()
    if not dados:
        raise HTTPException(status_code=422, detail="Arquivo vazio.")

    # Aceita vídeo ou imagem de frame
    content_type = arquivo.content_type or ""
    if not (content_type.startswith("video/") or content_type.startswith("image/")):
        raise HTTPException(
            status_code=422,
            detail="Envie um arquivo de vídeo (MP4, MOV, AVI) ou um frame de imagem.",
        )

    data_uri = frame_para_data_uri(dados)
    conteudo_hash = hashlib.md5(dados).hexdigest()

    try:
        client = replicate.Client(api_token=REPLICATE_API_TOKEN)
        saida = client.run(
            MODELO_IMAGEM,
            input={"image": data_uri, "prompt": PROMPT_VIDEO, "max_tokens": 1024, "temperature": 0.2},
        )
        descricao = "".join(saida).strip()
    except replicate.exceptions.ReplicateError as e:
        raise HTTPException(status_code=502, detail=f"Erro na API Replicate: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

    registro = models.Descricao(
        usuario_id=usuario.id,
        tipo=models.TipoDescricao.video,
        conteudo_hash=conteudo_hash,
        descricao=descricao,
        modelo=MODELO_IMAGEM,
        formato_original=content_type,
    )
    db.add(registro)
    db.commit()
    db.refresh(registro)

    return {
        "id": registro.id,
        "descricao": descricao,
        "modelo": MODELO_IMAGEM,
        "tipo": "video",
        "created_at": registro.created_at,
    }


@router.websocket("/aovivo")
async def interacao_aovivo(websocket: WebSocket, token: str):
    """
    WebSocket para interação ao vivo com a câmera.

    O cliente deve:
    1. Conectar com ?token=<jwt>
    2. Enviar frames como base64 (string JSON: {"frame": "<base64>"})
    3. Receber descrições em tempo real como JSON: {"descricao": "..."}

    Para encerrar, fechar a conexão WebSocket.
    """
    # Valida token antes de aceitar a conexão
    try:
        payload = decodificar_token(token)
        usuario_id = payload.get("sub")
        if not usuario_id:
            await websocket.close(code=4001)
            return
    except Exception:
        await websocket.close(code=4001)
        return

    await websocket.accept()

    try:
        client = replicate.Client(api_token=REPLICATE_API_TOKEN)
        while True:
            dados = await websocket.receive_json()
            frame_b64 = dados.get("frame")
            pergunta = dados.get("pergunta")

            if not frame_b64:
                await websocket.send_json({"erro": "Campo 'frame' obrigatório."})
                continue

            data_uri = f"data:image/jpeg;base64,{frame_b64}"
            prompt = pergunta if pergunta else PROMPT_AOVIVO

            try:
                saida = client.run(
                    MODELO_IMAGEM,
                    input={"image": data_uri, "prompt": prompt, "max_tokens": 256, "temperature": 0.1},
                )
                descricao = "".join(saida).strip()
                await websocket.send_json({"descricao": descricao})
            except Exception as e:
                await websocket.send_json({"erro": f"Erro ao processar frame: {str(e)}"})

    except WebSocketDisconnect:
        pass
