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
