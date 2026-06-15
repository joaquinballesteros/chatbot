# rag_tester.py
import os
import re
import asyncio
import streamlit as st
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_google_genai import  ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

from langchain_openai import OpenAIEmbeddings


from faiss_local import retrieve_mmr, retrieve_with_score, EMBEDDING_MODEL

INDEX_PATH_DEFAULT = "data/faiss_index"

# ------------- Utils -------------
def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

def highlight_terms(text: str, terms: List[str]) -> str:
    """Resalta términos (>=4 chars) en un bloque de código con markdown simple."""
    safe = text
    for t in terms:
        t = t.strip()
        if len(t) < 4:
            continue
        try:
            safe = re.sub(rf"(?i)\b({re.escape(t)})\b", r"**\1**", safe)
        except re.error:
            pass
    return safe

# ------------- App -------------
st.set_page_config(page_title="RAG Tester", layout="wide")
st.title("🔍 RAG Tester (FAISS + Google Embeddings)")
st.caption("App independiente para probar recuperación y contexto, usando la API key desde secrets.toml.")

with st.sidebar:
    st.header("⚙️ Configuración")
    index_path = st.text_input("Ruta del índice FAISS", value=INDEX_PATH_DEFAULT)
    use_llm = st.checkbox("Usar LLM para generar respuesta (Gemini)", value=False)
    temperature = st.slider("Temperatura LLM", 0.0, 1.0, 0.2, 0.05)
    st.markdown("---")
    st.markdown("### Método de recuperación")
    method = st.radio("Elige método", ["MMR (diversidad)", "Similarity + Score (distancia)"])
    k = st.number_input("k (resultados finales)", min_value=1, max_value=50, value=8)
    if method == "MMR (diversidad)":
        fetch_k = st.number_input("fetch_k (candidatos iniciales)", min_value=1, max_value=200, value=40)
        lambda_mult = st.slider("lambda_mult (0=relevancia, 1=diversidad)", 0.0, 1.0, 0.5, 0.05)
    else:
        fetch_k = st.number_input("k búsqueda principal", min_value=1, max_value=200, value=30)
        max_distance = st.text_input("Filtro max_distance (opcional, distancia FAISS)")

# Cargar embeddings + vectorstore
ensure_event_loop()

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error("❌ No se encontró GOOGLE_API_KEY en .streamlit/secrets.toml")
    st.stop()

embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=st.secrets["OPENAI_API_KEY"]
    )
if not os.path.exists(index_path):
    st.error(f"Índice no encontrado en: {index_path}")
    vectorstore = None
else:
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    st.sidebar.success(f"✅ Índice cargado: {index_path}")

query = st.text_input("Escribe una query para probar la recuperación…", "")
go = st.button("🚀 Probar recuperación", use_container_width=True)

def retrieve_mmr(q: str, k: int, fetch_k: int, lambda_mult: float):
    try:
        return vectorstore.max_marginal_relevance_search(q, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
    except Exception:
        return vectorstore.similarity_search(q, k=k)

def retrieve_with_score(q: str, k_final: int, fetch_k: int, max_dist: float | None):
    docs_scores = vectorstore.similarity_search_with_score(q, k=fetch_k)
    docs_scores.sort(key=lambda x: x[1])  # menor distancia primero
    if max_dist is not None:
        docs_scores = [ds for ds in docs_scores if ds[1] <= max_dist]
    top = docs_scores[:k_final]
    return [d for d, _ in top], top

if go and vectorstore and query.strip():
    st.subheader("📦 Resultados de recuperación")
    terms = [t for t in re.findall(r"[A-Za-z_]\w+", query) if len(t) >= 4]

    if method == "MMR (diversidad)":
        docs = retrieve_mmr(query, k, fetch_k, lambda_mult)
        st.info(f"MMR: k={k}, fetch_k={fetch_k}, lambda_mult={lambda_mult}")
        for i, d in enumerate(docs, start=1):
            m = d.metadata or {}
            st.markdown(
                f"**#{i}** — `{m.get('source','?')}` • pkg:`{m.get('package','')}` • kind:`{m.get('kind','')}` • "
                f"class:`{m.get('class','')}` • methods:`{m.get('methods','')}` • "
                f"enum:`{m.get('enum_constants','')}` • chunk:{m.get('chunk_id','')}"
            )
            st.code(highlight_terms(d.page_content[:2000], terms), language="java")
        context_text = "\n\n---\n\n".join([d.page_content for d in docs])

    else:
        try:
            md = float(max_distance) if max_distance.strip() else None
        except Exception:
            md = None
        docs, docs_scores = retrieve_with_score(query, k, fetch_k, md)
        st.info(f"Similarity+Score: fetch_k={fetch_k}, k_final={k}, max_distance={md}")
        with st.expander("🔢 Distancias (menor = mejor)"):
            for rank, (d, dist) in enumerate(docs_scores, start=1):
                m = d.metadata or {}
                st.markdown(
                    f"**#{rank}** dist=`{dist:.6f}` — `{m.get('source','?')}` • pkg:`{m.get('package','')}` • class:`{m.get('class','')}` • kind:`{m.get('kind','')}`"
                )
        for i, d in enumerate(docs, start=1):
            m = d.metadata or {}
            st.markdown(
                f"**#{i}** — `{m.get('source','?')}` • pkg:`{m.get('package','')}` • kind:`{m.get('kind','')}` • "
                f"class:`{m.get('class','')}` • methods:`{m.get('methods','')}` • "
                f"enum:`{m.get('enum_constants','')}` • chunk:{m.get('chunk_id','')}"
            )
            st.code(highlight_terms(d.page_content[:2000], terms), language="java")
        context_text = "\n\n---\n\n".join([d.page_content for d in docs])

    # ---------- (Opcional) Responder con LLM ----------
    if use_llm:
        st.subheader("🧠 Respuesta con LLM (usando contexto recuperado)")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key, temperature=temperature)
        prompt_tpl = PromptTemplate.from_template("""
Eres un tutor de programación experto y tu objetivo es personalizar la asistencia basándote en el historial del estudiante para fomentar la innovación y el pensamiento crítico. No debes dar respuestas directas. Tu método se basa en guiar al estudiante hacia la solución.

** Posibles Fragmentos de Referencia (uso interno, NUNCA mostrar al estudiante):**
{context}

**Reglas Estrictas de Interacción:**
Usa el Método Socrático: nunca des la respuesta directa. Guía con preguntas.
Adapta la dificultad según el historial.
Fomenta la autoexplicación.
Da retroalimentación constructiva y personalizada.
Estimula la curiosidad: termina con preguntas abiertas.
Nunca muestres directamente el contenido de las Soluciones de Referencia. Solo utilízalas como apoyo para generar tus preguntas y orientaciones.
Implementación en Java o C según indique el estudiante.

**Pregunta Actual del Estudiante:**
{question}

**Respuesta del Tutor:**""")
        chain = prompt_tpl | llm
        resp = chain.invoke({"context": context_text, "question": query}).content
        st.markdown(resp)
