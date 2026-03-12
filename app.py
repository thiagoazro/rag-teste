
from __future__ import annotations

import os
from pathlib import Path

import faiss
import pandas as pd
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_store():
    idx_path = DATA_DIR / "faiss.index"
    meta_path = DATA_DIR / "meta.parquet"
    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Rode `python build_index.py` antes.")
    index = faiss.read_index(str(idx_path))
    meta = pd.read_parquet(meta_path)
    return index, meta


@st.cache_resource
def load_llm_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def retrieve(query: str, k: int, source_filter: str | None = None):
    model = load_embedding_model()
    index, meta = load_store()

    if source_filter and source_filter != "todas":
        meta_f = meta[meta["source"] == source_filter].reset_index(drop=False)
        if meta_f.empty:
            return []
        texts = (meta_f["title"].fillna("") + "\n\n" + meta_f["text"].fillna("")).tolist()
        emb = model.encode(texts, normalize_embeddings=True).astype("float32")
        tmp = faiss.IndexFlatIP(emb.shape[1])
        tmp.add(emb)
        q = model.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = tmp.search(q, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            row = meta_f.iloc[int(idx)].to_dict()
            row["score"] = float(score)
            results.append(row)
        return results

    q = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        row = meta.iloc[int(idx)].to_dict()
        row["score"] = float(score)
        results.append(row)
    return results


def build_context(hits: list[dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Fonte {i}]",
                    f"doc_id: {hit.get('doc_id')}",
                    f"categoria: {hit.get('source')}",
                    f"titulo: {hit.get('title', '(sem título)')}",
                    f"conteudo: {hit.get('text', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def generate_rag_answer(query: str, hits: list[dict], llm_model: str) -> str:
    client = load_llm_client()
    if client is None:
        return ""

    context = build_context(hits)
    system_prompt = (
        "Voce responde perguntas de alunos com base exclusiva no contexto recuperado do FAQ. "
        "Se a resposta nao estiver no contexto, diga claramente que a base nao contem informacao suficiente. "
        "Seja objetivo, em portugues do Brasil, e cite as fontes como [Fonte 1], [Fonte 2]."
    )
    user_prompt = (
        f"Pergunta do aluno:\n{query}\n\n"
        f"Contexto recuperado:\n{context}\n\n"
        "Gere uma resposta final curta, fiel ao contexto, seguida de uma linha 'Fontes usadas:' "
        "com as referencias citadas."
    )
    response = client.chat.completions.create(
        model=llm_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def heuristic_answer(hits: list[dict]) -> str:
    unique_answers = []
    seen = set()

    for hit in hits:
        text = str(hit.get("text", "")).strip()
        if not text:
            continue

        answer_part = text
        marker = "**Resposta:**"
        if marker in text:
            answer_part = text.split(marker, 1)[1].strip()

        answer_part = " ".join(answer_part.split())
        if answer_part and answer_part not in seen:
            seen.add(answer_part)
            unique_answers.append(answer_part)

    if not unique_answers:
        return "Nao encontrei informacao suficiente na base para responder."

    if len(unique_answers) == 1:
        return unique_answers[0]

    return " ".join(unique_answers)


st.set_page_config(page_title="FAQ - empregadados (RAG)", layout="wide")
st.title("FAQ - empregadados — Busca Semântica + RAG")
st.caption(
    "O app recupera trechos com embeddings + FAISS e pode gerar uma resposta final com LLM usando apenas o contexto encontrado."
)

index, meta = load_store()
sources = ["todas"] + sorted([s for s in meta["source"].dropna().unique().tolist() if str(s).strip()])
llm_available = load_llm_client() is not None

with st.sidebar:
    st.header("Config")
    k = st.slider("Top-K", 3, 15, 5)
    source_filter = st.selectbox("Filtrar por categoria (source)", sources, index=0)
    use_llm = st.toggle("Gerar resposta com LLM (RAG)", value=llm_available, disabled=not llm_available)
    llm_model = st.text_input("Modelo do LLM", value=DEFAULT_LLM_MODEL, disabled=not llm_available)
    st.caption(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    if llm_available:
        st.success("OPENAI_API_KEY detectada. O modo RAG com geracao esta disponivel.")
    else:
        st.info("Configure OPENAI_API_KEY para habilitar a resposta final com LLM.")

q = st.text_input(
    "Pergunta:",
    placeholder="Ex.: como funciona reembolso? qual SLA do suporte? LGPD exclusao de dados?",
)

if q:
    hits = retrieve(q, k=k, source_filter=source_filter)

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader("Contexto recuperado")
        if not hits:
            st.warning("Nenhum resultado encontrado para esse filtro. Tente 'todas' ou outra categoria.")
        for i, h in enumerate(hits, 1):
            st.markdown(f"### {i}. {h.get('title', '(sem título)')} — score `{h['score']:.4f}`")
            st.caption(f"doc_id: {h.get('doc_id')} | categoria: {h.get('source')}")
            st.write(h.get("text", ""))
            st.divider()

    with right:
        st.subheader("Resposta final")
        if hits and use_llm and llm_available:
            with st.spinner("Gerando resposta com o LLM..."):
                answer = generate_rag_answer(q, hits, llm_model=llm_model)
            st.write(answer)
        elif hits:
            st.caption("Modo fallback sem LLM: concatenacao simples dos melhores trechos.")
            st.write(heuristic_answer(hits))
        else:
            st.info("Sem contexto recuperado, nao ha resposta para gerar.")
else:
    st.info(
        "Exemplos: 'como solicitar reembolso?', 'como emitir nota fiscal?', 'como ativar 2FA?', 'qual SLA do suporte?'"
    )
