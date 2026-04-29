# Pacote de correções — Descritoria

Este pacote contém arquivos substitutos para aplicar as mudanças solicitadas no app.

## Arquivos incluídos

- `models.py`
- `schemas.py`
- `routers/imagens.py`
- `routers/feedback.py`
- `routers/usuario.py`
- `routers/videos.py`
- `static/index.html`
- `scripts/migrate_user_learning.py`

## O que foi corrigido

1. Aba Câmera:
   - remove a opção de gerar 3 descrições;
   - retorna sempre uma única descrição;
   - coloca pergunta logo abaixo da descrição;
   - mantém feedback sempre após cada descrição;
   - usa prompt de zero alucinação;
   - separa botão de galeria do botão de câmera.

2. Aba Internet:
   - baixa a imagem no backend antes de enviar à IA;
   - valida bytes com Pillow;
   - converte formatos incompatíveis para JPEG;
   - melhora extração de imagens de páginas;
   - retorna erro acessível quando o site bloqueia download.

3. Pessoas:
   - remove a aba Pessoas da navegação principal;
   - coloca Pessoas dentro de Conta.

4. Documento:
   - troca um botão confuso por três botões separados:
     - tirar foto do documento;
     - escolher imagem da galeria;
     - escolher arquivo.

5. Ao Vivo:
   - implementa WebSocket funcional para frames;
   - faz a aba trabalhar por voz;
   - adiciona reconhecimento de voz quando disponível no navegador;
   - adiciona fala automática das descrições.

6. Conta:
   - melhora "Meu Estilo de Descrição";
   - adiciona perfil aprendido pela IA;
   - feedback positivo e negativo passam a alimentar esse perfil.

## Como aplicar

Na raiz do repositório:

```bash
cp -R descritoria_correcoes/* .
python scripts/migrate_user_learning.py
python -m compileall .
uvicorn main:app --reload
```

## Atenção

Se o app estiver em produção ou com usuários reais, faça backup do banco antes:

```bash
cp descritoria.db descritoria.db.backup
```

## Checklist de teste

- Abrir aba Câmera.
- Conferir que não existe mais "3 descrições".
- Escolher imagem da galeria.
- Tirar foto.
- Conferir se aparece uma descrição.
- Fazer pergunta logo abaixo da descrição.
- Enviar feedback bom e ruim.
- Verificar se a Conta mostra o perfil aprendido.
- Descrever URL de imagem.
- Descrever página inteira.
- Abrir Documento e testar os 3 botões.
- Iniciar Ao Vivo e confirmar fala automática.
