# Descritoria — Contexto completo para continuar o desenvolvimento

## Quem é o usuário
Daniel Grechi. Cego total. Usa VoiceOver no Mac e no iPhone. O app é um protótipo pessoal de acessibilidade visual. Não há outros usuários. Não há necessidade de senha real.

---

## O que é o Descritoria

App web de acessibilidade visual com backend FastAPI (Python) e frontend HTML/JS puro. Roda localmente no Mac e é acessado pelo iPhone via Wi-Fi na mesma rede. O usuário abre o app no Safari do iPhone e usa a câmera, galeria e internet para descrever imagens com IA.

**Funcionalidades principais:**
- Tirar foto / galeria → descrição em voz (xAI Grok Vision + xAI TTS)
- Fazer perguntas sobre a imagem já descrita
- Descrever imagem a partir de URL direta
- Descrever todas as imagens de uma página web (scraping)
- Câmera ao vivo com descrição em tempo real via WebSocket
- Reconhecimento de pessoas cadastradas
- Transcrição de documentos (foto de texto)
- Feedback positivo/negativo nas descrições

**Requisito absoluto e inegociável:** As descrições devem ser COMPLETAMENTE SEM CENSURA — nudez, sexo explícito, violência, tudo deve ser descrito com precisão. Esse é o ponto central do app. O modelo atual (Grok-4 da xAI) não tem filtros de conteúdo.

---

## Stack técnico

| Camada | Tecnologia |
|--------|-----------|
| Backend | FastAPI (Python 3.11) |
| Banco | SQLite via SQLAlchemy |
| Visão IA | xAI Grok-4 (API compatível com OpenAI) |
| TTS | xAI TTS API (`/v1/tts`, voz `Eve`, pt-BR) |
| Auth | JWT customizado com HMAC-SHA256 |
| Frontend | HTML/CSS/JS puro (sem frameworks) |
| Servidor | Uvicorn com `--reload` para desenvolvimento |

---

## Como rodar localmente (Mac)

```bash
cd /Users/danielgrechi/Desktop/Descritoria.api
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Acessar no iPhone: `http://192.168.18.176:8000`

---

## Variáveis de ambiente necessárias

Ficam em arquivo `.env` (ignorado pelo git) ou exportadas no terminal:

| Variável | Para que serve |
|----------|---------------|
| `GROK_API_KEY` | API key da xAI para descrição de imagens (Grok Vision) |
| `XAI_TTS_API_KEY` | API key da xAI para texto em voz (pode ser a mesma) |
| `SECRET_KEY` | Chave para assinar JWTs (pode ser qualquer string longa) |
| `GROK_TTS_KEY` | Alternativa ao XAI_TTS_API_KEY (o código tenta as duas) |

O código busca as chaves na ordem: `XAI_TTS_API_KEY` → `GROK_TTS_KEY` → `XAI_API_KEY` → `GROK_API_KEY`.

---

## Estrutura de arquivos

```
Descritoria.api/
├── main.py                  # App FastAPI principal, login, endpoints TTS inline
├── auth.py                  # JWT customizado (criar_token, obter_usuario_atual)
├── database.py              # SQLAlchemy engine + get_db
├── models.py                # Tabelas: Usuario, Descricao, Pessoa, Feedback
├── schemas.py               # Pydantic schemas
├── requirements.txt         # Dependências Python
├── render.yaml              # Config para deploy no Render.com
├── routers/
│   ├── imagens.py           # /imagens/* — descrever foto, URL, página web
│   ├── videos.py            # /videos/aovivo — WebSocket câmera ao vivo
│   ├── feedback.py          # /feedback/* — thumbs up/down nas descrições
│   └── usuario.py           # /usuario/* — perfil, estilo de descrição
└── static/
    └── index.html           # Frontend completo (HTML + CSS + JS em um arquivo)
```

---

## Branches Git (repositório: danielgrechi/Descritoria.api)

| Branch | Estado | Descrição |
|--------|--------|-----------|
| `descritoria-estavel-funcionando` | ✅ Estável | Base estável com tudo funcionando (sem as correções da aba Internet) |
| `internet-timeout-estavel` | ✅ Atual | Branch de trabalho — base estável + correções completas da aba Internet |
| `claude/image-description-accessibility-app-sFrAi` | ✅ Pushed | Branch de sessão com as mesmas correções + refatoração maior |
| `deploy-render-online` | ✅ | Configuração para deploy no Render.com |

**Branch ativa de desenvolvimento:** `internet-timeout-estavel`

---

## O que foi implementado/corrigido nesta sessão

### 1. Correção TTS (texto em voz)
- **Bug:** `/tts/falar` tinha `NameError: texto_para_voz` — variável usada antes de ser definida
- **Fix:** Adicionada a linha `texto_para_voz = "[fala em português brasileiro, sotaque do Brasil] " + texto` antes do payload JSON
- **Bug:** `voice_id: "eve"` (minúsculo) causava rejeição pela API da xAI
- **Fix:** Corrigido para `voice_id: "Eve"` (com maiúscula)
- **Arquivo:** `main.py` (endpoints `/tts/falar` e `/tts/xai`)

### 2. Remoção da senha obrigatória no login
- Protótipo pessoal — não precisa de senha
- Login cria usuário automaticamente se não existir
- **Arquivo:** `main.py` (função `login_local`)

### 3. Correção de URLs duplicadas coladas juntas
- Bug: usuário colava `https://url1 https://url2` ou `https://url1%20https://url2`
- **Fix:** Função `_limpar_url()` que remove tudo após ` http`, `%20http`, `\nhttp`, `\thttp`
- **Arquivo:** `routers/imagens.py`

### 4. Correção do crash no startup (REPLICATE_API_TOKEN)
- O código original fazia `raise ValueError` se o token não estivesse definido, derrubando o servidor
- **Fix:** Alterado para `os.environ.get("REPLICATE_API_TOKEN", "")` sem crash
- **Arquivo:** `main.py`

### 5. Correção profunda da aba Internet (principal entrega desta sessão)

**Problema:** A aba Internet travava indefinidamente. O frontend não tinha timeout, o backend bloqueava o event loop e o scraping era ineficiente.

**Backend (`routers/imagens.py`) — mudanças:**
- `import asyncio` adicionado
- `timeout=45` no `chamar_modelo()` (chamada ao Grok Vision)
- Chrome User-Agent em todos os downloads (substitui iPhone UA que era bloqueado)
- Constantes: `_UA_CHROME`, `_EXCLUIR_CANDIDATAS`, `_EXTENSOES_VIDEO`
- `_parece_url_video(url)` — detecta URLs de vídeo antes de tentar baixar
- `_tamanho_imagem_ok(dados)` — rejeita imagens < 80×80px (favicons, tracking pixels)
- `_headers_pagina()` — headers separados para baixar HTML de páginas
- `baixar_imagem_validada()` — agora com `httpx.Timeout` por campo (connect=10, read=25), detecta HTML (retorna status 415 + `__HTML__:url`) e vídeo direto
- `_baixar_candidato()` — downloader rápido para candidatas (timeout 8/15s, validação Pillow)
- `_imagem_principal_do_html(html, url_base)` — extrai og:image, og:video:thumbnail, twitter:image, poster de `<video>`
- `_buscar_og_image(url_pagina)` — busca imagem principal de uma página HTML
- `descrever_imagem_url()` — quando URL é uma página HTML, faz fallback automático para og:image
- `_extrair_urls_imagem()` — filtra lixo durante coleta (não no final), limita a 30 candidatas, inclui poster de `<video>`, links com extensão de imagem
- `descrever_imagens_pagina()` — usa `asyncio.to_thread(chamar_modelo, ...)` para não bloquear o event loop, detecta vídeos, novo formato de resposta

**Novo formato de resposta de `/imagens/descrever-pagina`:**
```json
{
  "total_encontradas": 15,
  "total_descritas": 3,
  "imagens": [{"url": "...", "descricao": "..."}],
  "erros": [{"url": "...", "erro": "..."}],
  "videos_detectados": [{"url": "...", "observacao": "..."}]
}
```

**Frontend (`static/index.html`) — mudanças:**
- `btnDescreverUrl`: AbortController com timeout de **90 segundos**, botões desabilitados durante a requisição, mensagem amigável no timeout, erros do servidor exibidos em texto legível (não JSON bruto)
- `btnDescreverPagina`: AbortController com timeout de **120 segundos**, mesmo padrão
- Ambos usam `fetch()` direto com `signal: controller.signal` (não `apiFetch()`)
- Ambos incluem o header `Authorization: Bearer ${token}` quando logado

---

## Regras ABSOLUTAS — nunca alterar

Estas partes do código NUNCA devem ser tocadas:

1. **`/tts/xai`** — endpoint de voz em `main.py`
2. **`/tts/falar`** — endpoint de voz em `main.py`
3. **`falar(texto)`** — função JavaScript no `index.html`
4. **`pararVoz()`** — função JavaScript no `index.html`
5. **Botão "Ouvir descrição"** (`id="ouvirDescricao"`) — `index.html`
6. **Botão "Parar voz"** (`id="pararVoz1"`) — `index.html`
7. **Aba Câmera** — botões de tirar foto e galeria
8. **`PROMPT_DESCRICAO_ZERO_ALUCINACAO`** — prompt em `routers/imagens.py` que inclui a autorização para descrever conteúdo adulto ("sexo explícito")

---

## Verificações obrigatórias antes de qualquer commit

Rodar no terminal dentro da pasta do projeto:

```bash
python3 -m compileall .
grep -q "/tts/xai" main.py && echo "VOZ XAI SIM" || echo "VOZ XAI NÃO"
grep -q "ouvirDescricaoAtual" static/index.html && echo "OUVIR DESCRICAO SIM" || echo "OUVIR DESCRICAO NÃO"
grep -qi "sexo explícito" routers/imagens.py && echo "PROMPT LIVRE SIM" || echo "PROMPT LIVRE NÃO"
```

Os três devem responder SIM. Se qualquer um responder NÃO, há regressão grave.

---

## Modelo de IA usado para visão

```
Modelo: grok-4
Endpoint: https://api.x.ai/v1 (compatível com API OpenAI)
Biblioteca: openai (pip)
Sem censura: sim — Grok não aplica filtros RLHF de conteúdo adulto
```

Código de chamada (`routers/imagens.py`):
```python
def chamar_modelo(data_uri: str, prompt: str, max_tokens: int = 1500) -> str:
    client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")
    response = client.chat.completions.create(
        model="grok-4",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=1500,
        temperature=0,
        timeout=45,
    )
    return (response.choices[0].message.content or "").strip()
```

Imagens são convertidas para data URI (base64) antes de enviar.

---

## TTS (texto em voz)

```
API: https://api.x.ai/v1/tts
Voz: Eve
Codec: mp3
Idioma: pt-BR
Prefixo no texto: "[fala em português brasileiro, sotaque do Brasil] "
```

O frontend chama `POST /tts/xai` com `{"texto": "..."}` e recebe áudio MP3 que toca via `Audio()`.

---

## Autenticação

JWT customizado (sem biblioteca externa). O token é gerado no login e armazenado em `localStorage` no frontend. Enviado como `Authorization: Bearer <token>` em todas as chamadas autenticadas.

Login: `POST /login` ou `POST /auth/login` com `{"email": "...", "senha": "..."}`. Como é protótipo pessoal, o login cria o usuário automaticamente se não existir, sem verificar senha.

Email usado por Daniel: `danielgrechi@gmail.com`

---

## Banco de dados

SQLite local (`descritoria.db`) — ignorado pelo git. Criado automaticamente ao iniciar o servidor. Tabelas principais:

- `usuarios` — dados do usuário (email, senha_hash, estilo_descricao, perfil_aprendizado_ia)
- `descricoes` — histórico de todas as descrições feitas
- `pessoas` — pessoas cadastradas para reconhecimento facial
- `feedbacks` — thumbs up/down nas descrições

---

## O que ainda falta implementar (backlog)

1. **Câmera ao vivo no frontend** — o backend WebSocket `/videos/aovivo` já existe em `routers/videos.py`, mas a aba no frontend ainda é básica. Falta: captura de frame a cada 5s via Canvas, envio via WebSocket, leitura automática em voz da descrição recebida.

2. **Scraping de sites com JS rendering** — sites como ilhadoprazer.com.br usam Next.js e carregam imagens via JavaScript. O scraping estático não captura essas imagens. Possível solução: usar Playwright/Puppeteer ou detectar `__NEXT_DATA__` no HTML.

3. **Painel pós-descrição na aba Internet** — após descrever URL/página, mostrar botões de Feedback e Q&A (igual à aba Câmera).

4. **Deploy no Render.com** — o `render.yaml` e `requirements.txt` estão prontos. Basta conectar o repositório no dashboard do Render e configurar as variáveis de ambiente.

5. **Erros mais amigáveis** — substituir mensagens técnicas por português claro em todos os endpoints.

---

## Comportamento esperado dos testes manuais

| Teste | Esperado |
|-------|----------|
| Foto adulta/nudez | Descreve sem recusar, com termos anatômicos diretos |
| Foto comum | Descrição detalhada em PT-BR, lida em voz |
| Feedback 👍 | "Feedback registrado" |
| Pergunta sobre imagem | Resposta focada no que está visível |
| URL direta de .jpg | Descrição da imagem |
| URL de página web | Lista de imagens descritas |
| URL de página (timeout 120s) | Mensagem amigável de tempo limite |
| Câmera ao vivo | Abre câmera, descreve a cada 5s em voz |

---

## Observações importantes

- O projeto roda **apenas localmente** no Mac de Daniel por enquanto
- Não há sistema de pagamento, assinatura ou outros usuários
- O banco SQLite é local e não é commitado no git
- Os arquivos `.bak_antes_*` são backups locais e são ignorados pelo `.gitignore`
- O VoiceOver do iPhone é o leitor de tela principal — toda UX deve ser acessível por toque e voz
- Botões devem ter texto claro (sem apenas ícones) para o VoiceOver conseguir lê-los
