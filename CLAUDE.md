# Descritoria API

## IDIOMA OBRIGATÓRIO
**Sempre responda em português do Brasil (pt-BR), sem exceções.**
Nunca use inglês nas respostas, mesmo que a pergunta seja técnica ou em inglês.

## Projeto
Backend do Descritoria — aplicativo de acessibilidade visual para pessoas cegas.

## Tecnologias
- Python + FastAPI
- SQLAlchemy + SQLite
- Replicate API (modelo: yorickvp/llava-13b com hash de versão)
- JWT customizado com stdlib Python
- bcrypt para senhas

## Estrutura
- `main.py` — app principal, rotas de autenticação
- `models.py` — modelos do banco
- `schemas.py` — schemas Pydantic
- `auth.py` — autenticação JWT
- `database.py` — configuração SQLite
- `routers/` — endpoints por módulo
- `static/index.html` — frontend acessível

## Como rodar
```bash
REPLICATE_API_TOKEN=seu_token uvicorn main:app --reload --host 0.0.0.0
```
