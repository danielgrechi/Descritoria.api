from fastapi import APIRouter

# As rotas de internet estão em imagens.py: /imagens/descrever-url e /imagens/descrever-pagina
router = APIRouter(prefix="/internet", tags=["Internet"])
