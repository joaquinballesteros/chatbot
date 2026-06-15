import os
import asyncio
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# === CONFIG ===
JAVA_ROOT = "repo/src/"   # <--- pon aquí la raíz del repo
INDEX_PATH = "data/faiss_index"              # mismo path que usa tu app
MAX_CHUNK_SIZE = 2000                   # tokens (aprox)
CHUNK_OVERLAP = 200                     # tokens (aprox)
EMBEDDING_MODEL = "text-embedding-3-large"  # modelo de embedding de Google
MAX_INPUT_TOKENS = 4096                 # modelo de chat (genera menos tokens que embedding)
MAX_TOTAL_TOKENS = 8192                 # modelo de chat (input + output)
MAX_OUTPUT_TOKENS = MAX_TOTAL_TOKENS - MAX_INPUT_TOKENS
TOP_K = 8                               # nº de chunks a recuperar
FETCH_K = 40                           # nº de chunks a recuperar inicialmente (MMR o score)
LAMBDA_MMR = 0.5                        # trade-off relevancia/diversidad
MAX_DISTANCE = None                     # filtrar por distancia (None = no filtrar)
# =================

@st.cache_resource
def inicializar_vectorstore(api_key: str):
    if not os.path.exists(INDEX_PATH):
        st.error(f"Índice vectorial '{INDEX_PATH}' no encontrado.")
        return None
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    st.sidebar.success("✅ Índice RAG (OpenAI embeddings) cargado.", icon="📚")
    return vectorstore

# --- RETRIEVAL (RAG) PARA CÓDIGO JAVA ---
# Opción A: MMR (diversidad). Muy recomendable para preguntas ambiguas.
def retrieve_mmr(vectorstore, query: str, k: int = TOP_K, fetch_k: int = FETCH_K, lambda_mult: float = LAMBDA_MMR):
    """
    Devuelve documentos usando Max Marginal Relevance (diversidad).
    - k: nº final de chunks
    - fetch_k: nº inicial de candidatos
    - lambda_mult: trade-off relevancia/diversidad (0..1)
    """
    try:
        docs = vectorstore.max_marginal_relevance_search(
            query=query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )
    except Exception:
        # Fallback si el método no está disponible en tu versión
        docs = vectorstore.similarity_search(query, k=k)
    return docs

# Opción B: con score (distancia). Menor = mejor en FAISS.
def retrieve_with_score(vectorstore, query: str, k: int = TOP_K, fetch_k: int = FETCH_K, max_distance: float = MAX_DISTANCE):
    """
    Recupera con score de FAISS (distancia). Ordena ascendente (mejor primero).
    - max_distance: si se da, filtra resultados con distancia <= max_distance.
    """
    docs_scores = vectorstore.similarity_search_with_score(query, k=fetch_k)
    # docs_scores: List[Tuple[Document, float]]  (float = distancia)
    docs_scores.sort(key=lambda x: x[1])  # menor distancia primero
    if max_distance is not None:
        docs_scores = [ds for ds in docs_scores if ds[1] <= max_distance]
    docs = [d for d, _ in docs_scores[:k]]
    return docs, docs_scores[:k]
