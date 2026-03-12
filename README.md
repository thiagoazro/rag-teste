# FAQ - empregadados

Projeto didático para demonstrar um pipeline de FAQ com:

- Python
- `sentence-transformers` para embeddings
- FAISS para índice vetorial local
- Streamlit para interface
- OpenAI API para resposta final com LLM sobre o contexto recuperado

## Arquitetura atual

1. Lê `docs.csv` com perguntas e respostas.
2. Gera embeddings para cada linha.
3. Salva o índice vetorial em FAISS.
4. Recupera os trechos mais relevantes no app.
5. Opcionalmente envia a pergunta + contexto recuperado para um LLM e gera a resposta final com RAG.

## Estrutura

```text
FAQ-empregadados/
  app.py
  build_index.py
  docs.csv
  requirements.txt
  render.yaml
  FAQ-empregadados-explicado-v2.ipynb
  streamlit_rag_render_aula.ipynb
  data/
    faiss.index
    meta.parquet
```

## Como rodar localmente

### 1) Instalar dependências

```bash
pip install -r requirements.txt
```

### 2) Construir o índice vetorial

```bash
python build_index.py
```

Arquivos gerados:

- `data/faiss.index`
- `data/meta.parquet`

### 3) Definir a chave da API do LLM

Linux/macOS:

```bash
export OPENAI_API_KEY="sua-chave"
export OPENAI_MODEL="gpt-4o-mini"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="sua-chave"
$env:OPENAI_MODEL="gpt-4o-mini"
```

Se `OPENAI_API_KEY` nao estiver configurada, o app continua funcionando apenas com recuperacao semantica.

### 4) Rodar a interface

```bash
streamlit run app.py
```

## Dataset

O arquivo `docs.csv` usa as colunas:

- `doc_id`: identificador do registro
- `title`: titulo resumido da pergunta
- `text`: pergunta e resposta
- `source`: categoria, como `financeiro`, `suporte`, `lgpd`

## O que mudou com o RAG

Antes:

- O app apenas mostrava os Top-K resultados e concatenava alguns trechos.

Agora:

- O app separa recuperacao e geracao.
- O LLM recebe somente os trechos recuperados.
- A resposta final pede citacoes como `[Fonte 1]`, `[Fonte 2]`.
- Sem chave de API, o sistema cai em fallback sem quebrar a aula.

## Sugestao de roteiro para aula

1. Mostrar o `build_index.py` e explicar embeddings.
2. Mostrar o `retrieve()` no `app.py`.
3. Perguntar algo no app e analisar os Top-K.
4. Habilitar o LLM e comparar "busca pura" vs "RAG".
5. Alterar o `docs.csv`, regerar o indice e repetir a consulta.

## Deploy no Render

O projeto inclui `render.yaml`, com start command do Streamlit.
Tambem inclui `.python-version` para forcar Python 3.11.11 no Render e evitar erro de build do `pandas`.

No painel do Render:

1. Criar um novo `Web Service` a partir do repositório.
2. Confirmar o ambiente Python.
3. Validar o `Build Command`: `pip install -r requirements.txt`
4. Validar o `Start Command`: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`
5. Adicionar a variavel `OPENAI_API_KEY`.
6. Opcionalmente adicionar `OPENAI_MODEL`.

## Observacao didatica

Para manter a aula simples:

- cada linha do CSV continua sendo um documento
- nao ha chunking
- nao ha reranking
- o RAG usa apenas contexto do FAQ

Isso e suficiente para mostrar a diferenca entre:

- busca vetorial
- pipeline RAG
- aplicacao web no Streamlit
- deploy em nuvem
# rag-teste
# rag-teste
# rag-teste
