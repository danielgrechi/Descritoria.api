from __future__ import annotations
import base64
import os
import time

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from openai import OpenAI, OpenAIError

from auth import decodificar_token

router = APIRouter(prefix="/videos", tags=["Vídeos e Ao Vivo"])

GROK_API_KEY = os.environ.get("GROK_API_KEY", "")
MODELO_IMAGEM = os.environ.get("GROK_VISION_MODEL", "grok-4")

PROMPT_AO_VIVO = """
Você é um assistente de acessibilidade visual ao vivo para pessoas cegas.
Descreva em português do Brasil somente o que está visível no frame atual.
Não invente objetos, textos, pessoas, emoções, riscos ou contexto externo.
Se o usuário fez uma pergunta, responda apenas com base no frame.
Se não houver certeza, diga "não é possível confirmar pela imagem".
A resposta deve ser curta, falável em voz alta e útil para orientação imediata.
""".strip()


def _grok_client() -> OpenAI:
    if not GROK_API_KEY:
        raise HTTPException(status_code=500, detail="GROK_API_KEY não configurada no servidor.")
    return OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")


def _validar_token_query(token: str) -> int:
    try:
        payload = decodificar_token(token)
        return int(payload.get("sub"))
    except Exception:
        raise HTTPException(status_code=401, detail="Token inválido.")


def _descrever_frame(data_uri: str, pergunta: str | None = None) -> str:
    prompt = PROMPT_AO_VIVO
    if pergunta:
        prompt += f"\n\nPergunta do usuário: {pergunta.strip()[:500]}"

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
        max_tokens=350,
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


@router.websocket("/aovivo")
async def camera_ao_vivo(websocket: WebSocket, token: str = Query(...)):
    try:
        _validar_token_query(token)
    except HTTPException:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    await websocket.send_json({"tipo": "status", "mensagem": "Câmera ao vivo conectada."})

    ultimo_processamento = 0.0
    intervalo_minimo = 4.0

    try:
        while True:
            payload = await websocket.receive_json()
            tipo = payload.get("tipo")

            if tipo == "ping":
                await websocket.send_json({"tipo": "status", "mensagem": "Conexão ativa."})
                continue

            if tipo != "frame":
                await websocket.send_json({"tipo": "erro", "mensagem": "Mensagem inválida. Envie tipo='frame'."})
                continue

            agora = time.monotonic()
            if agora - ultimo_processamento < intervalo_minimo:
                await websocket.send_json({"tipo": "ignorado", "mensagem": "Frame ignorado para reduzir custo e evitar excesso de chamadas."})
                continue
            ultimo_processamento = agora

            data_uri = payload.get("data_uri")
            pergunta = payload.get("pergunta")

            if not data_uri or not isinstance(data_uri, str) or not data_uri.startswith("data:image/"):
                await websocket.send_json({"tipo": "erro", "mensagem": "Frame inválido."})
                continue

            # Limite aproximado para proteger memória/custo.
            if len(data_uri) > 4_000_000:
                await websocket.send_json({"tipo": "erro", "mensagem": "Frame muito grande. Reduza a resolução."})
                continue

            try:
                descricao = _descrever_frame(data_uri, pergunta)
                await websocket.send_json({"tipo": "descricao", "texto": descricao})
            except OpenAIError as e:
                await websocket.send_json({"tipo": "erro", "mensagem": f"Erro na API Grok: {str(e)}"})
            except Exception as e:
                await websocket.send_json({"tipo": "erro", "mensagem": f"Erro ao descrever frame: {str(e)}"})

    except WebSocketDisconnect:
        return
