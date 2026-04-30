# Deploy do Descritoria no Render

## O que você vai precisar

- Conta gratuita no [Render](https://render.com)
- O repositório do projeto no GitHub (`danielgrechi/Descritoria.api`)
- Suas chaves de API (xAI e Replicate)

---

## Passo a passo

### 1. Acesse o Render

Entre em https://render.com e faça login (pode usar a conta Google).

### 2. Crie um novo Web Service

- Clique em **New +** → **Web Service**
- Escolha **Connect a repository**
- Selecione o repositório `danielgrechi/Descritoria.api`
- Se não aparecer, clique em **Configure GitHub** e autorize o Render

### 3. Configure o serviço

Preencha os campos assim:

| Campo | Valor |
|---|---|
| **Name** | descritoria-api |
| **Region** | Oregon (US West) ou São Paulo (mais rápido para BR) |
| **Branch** | `descritoria-estavel-funcionando` |
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | Free (gratuito) |

### 4. Configure as variáveis de ambiente

Na seção **Environment Variables**, adicione cada uma dessas:

| Variável | Descrição | Obrigatória |
|---|---|---|
| `GROK_API_KEY` | Chave da API xAI para descrição de imagens | **Sim** |
| `XAI_API_KEY` | Mesma chave xAI (alias alternativo) | Recomendado |
| `XAI_TTS_API_KEY` | Chave xAI para voz (TTS) | **Sim** |
| `REPLICATE_API_TOKEN` | Token do Replicate (endpoint legado /describe) | Opcional |
| `SECRET_KEY` | Chave secreta para assinatura dos tokens JWT | **Sim** |

**Como preencher SECRET_KEY:**
Gere uma string aleatória segura, por exemplo:
```
descritoria-prod-2024-mude-esta-chave-para-algo-secreto
```

**NUNCA coloque as chaves no código ou em arquivos do repositório.**

### 5. Clique em Create Web Service

O Render vai:
1. Baixar o código do GitHub
2. Instalar as dependências (`pip install -r requirements.txt`)
3. Iniciar o servidor (`uvicorn main:app --host 0.0.0.0 --port $PORT`)

Aguarde o status ficar verde: **Live**.

### 6. Acesse o app

O Render vai gerar uma URL no formato:
```
https://descritoria-api.onrender.com
```

Abra essa URL no iPhone — o app vai funcionar de qualquer rede, sem precisar do Mac ligado.

---

## Banco de dados (SQLite)

O app usa SQLite local (`descritoria.db`) que funciona bem para uso pessoal.

**Limitação do plano gratuito do Render:**
No plano Free, o disco é efêmero — o banco de dados é apagado toda vez que o servidor reinicia (a cada deploy ou após inatividade).

**Para não perder dados entre reinicializações:**
- Use o Render Disk (US$ 1/mês) — um volume persistente
- Ou migre para PostgreSQL (o Render oferece plano gratuito com banco persistente)

Por ora, o SQLite funciona normalmente para testes. Se perder o banco, o app recria automaticamente e você faz login de novo.

---

## Comandos locais (Mac)

Para continuar testando no Mac normalmente:

```bash
cd ~/Desktop/Descritoria.api
sh iniciar.sh
```

O comando de produção que o Render usa:
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## Atualizar o app no Render

Sempre que fizer um novo commit na branch `descritoria-estavel-funcionando` e fizer push para o GitHub, o Render detecta automaticamente e faz novo deploy.

```bash
git push origin descritoria-estavel-funcionando
```

---

## Troubleshooting

**App não inicia (erro no log do Render):**
- Verifique se todas as variáveis de ambiente estão preenchidas no painel do Render
- Confira os logs em **Logs** → **Build** e **Logs** → **Live**

**Voz não funciona:**
- Confirme que `XAI_TTS_API_KEY` está preenchida com a chave correta

**Descrição de imagem não funciona:**
- Confirme que `GROK_API_KEY` está preenchida

**App lento na primeira abertura:**
- Normal no plano Free — o Render "hiberna" serviços inativos por 15 minutos
- Ao abrir, aguarde ~30 segundos para "acordar"
- Planos pagos não hibernam
