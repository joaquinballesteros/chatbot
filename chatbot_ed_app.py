# == chatbot_ed_app.py (Versión Final v7 - Con control de Reset) ==
import os
import streamlit as st
import pandas as pd
from datetime import datetime
import asyncio
import traceback

# --- LIBRERÍAS DE IA (LangChain y Google) ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# --- LIBRERÍAS DE FIREBASE ---
import google.oauth2.service_account
from google.cloud import firestore
from cryptography.fernet import Fernet # Necesario para cargar datos

# --- RUTAS Y CONSTANTES ---
TMP_DIR = "tmp"
INDEX_PATH = "faiss_index"
USERS_CSV_PATH_ENC = "data/users.csv.encrypted"

if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)
USERS_CSV_PATH_TMP = os.path.join(TMP_DIR, "users.csv.tmp")

# --- CONEXIÓN A FIREBASE ---
@st.cache_resource
def get_firestore_client():
    key_dict = st.secrets["firebase_service_account"]
    creds = google.oauth2.service_account.Credentials.from_service_account_info(key_dict)
    db = firestore.Client(credentials=creds)
    return db

# --- GESTIÓN DE HISTORIAL ---
def load_active_user_history(db: firestore.Client, user_id: str):
    historial_ref = db.collection('users').document(user_id).collection('historial')
    reset_query = historial_ref.where("role", "==", "system").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1)
    reset_docs = list(reset_query.stream())
    last_reset_timestamp = reset_docs[0].get('timestamp') if reset_docs else None
    query = historial_ref.where("timestamp", ">", last_reset_timestamp).order_by("timestamp") if last_reset_timestamp else historial_ref.order_by("timestamp")
    return [doc.to_dict() for doc in query.stream()]

def save_message_to_history(db: firestore.Client, user_id: str, message: dict):
    message_to_save = message.copy()
    message_to_save['timestamp'] = firestore.SERVER_TIMESTAMP
    db.collection('users').document(user_id).collection('historial').add(message_to_save)

def reset_user_chat(db: firestore.Client, user_id: str):
    reset_message = {"role": "system", "content": f"--- CHAT RESETEADO POR EL USUARIO ---"}
    save_message_to_history(db, user_id, reset_message)
    st.session_state.messages = []

# --- FUNCIONES AUXILIARES ---
@st.cache_data(ttl=3600)
def cargar_datos_estudiantes():
    fernet_key = st.secrets["encryption"]["key"].encode()
    fernet = Fernet(fernet_key)
    with open(USERS_CSV_PATH_ENC, 'rb') as f_enc:
        token = f_enc.read()
    data = fernet.decrypt(token)
    with open(USERS_CSV_PATH_TMP, 'wb') as f_dec:
        f_dec.write(data)
    df = pd.read_csv(USERS_CSV_PATH_TMP, dtype=str)
    df['IDCV'] = df['IDCV'].str.strip()
    os.remove(USERS_CSV_PATH_TMP)
    return df

@st.cache_resource
def inicializar_vectorstore(api_key: str):
    if not os.path.exists(INDEX_PATH):
        st.error(f"Índice vectorial '{INDEX_PATH}' no encontrado.")
        return None
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    st.sidebar.success("✅ Índice RAG cargado.", icon="📚")
    return vectorstore

# --- LÓGICA DE RAG EN DOS PASOS ---
file_topic_mapping = {
    # Primeros ficheros de C, punteros y grafos
    "TipoPuntero.pdf": "Descripción de tipos de punteros en C",
    "Punteros_y_Arrays.pdf": "Relación entre punteros y arrays en C y aritmética de punteros",
    "MemoriaDinámicaParte1.pdf": "Conceptos de memoria dinámica en C: malloc, calloc, realloc y free",
    "MemoriaDinámicaParte2.pdf": "Modularidad en C y estructuras dinámicas: colas, pilas y listas enlazadas",
    "Basicos de C.pdf": "Conceptos básicos del lenguaje C: sintaxis, tipos de datos y E/S",
    "Sesion1GrafosR.pdf": "Fundamentos de grafos: definiciones, representaciones y propiedades",
    "Sesion2GrafosR.pdf": "Recorridos en grafos: profundidad, amplitud y algoritmos de recorrido",
    "RepasoJava.pdf": "Revisión del lenguaje Java: clases, objetos, memoria y registros",
    "Montículos Binarios (30 oct).pdf": "Montículos binarios: implementación, operaciones y orden del heap",
    # Ficheros de TADs y estructuras en Java/C
    "ListaEnlazadaPriorizada.pdf": "Módulo en C para gestión de lista enlazada de canales favoritos ordenados: crear, destruir, mostrar e insertar",
    "ADT_Colas.pdf": "Definición e implementación del TAD Cola en Java",
    "ADT_Listas.pdf": "Definición e implementación del TAD Lista en Java",
    "ADT_Pilas.pdf": "Definición e implementación del TAD Pila en Java",
    "ADT_Set.pdf": "Definición e implementación del TAD Conjunto en Java",
    "Árboles.pdf": "Introducción a árboles genéricos: terminología, recorridos y representación en Java",
    "BST.pdf": "Árboles binarios de búsqueda: propiedades y operaciones de búsqueda, inserción y eliminación",
    "Hash (25 nov) 1 de 2 .pdf": "Fundamentos de tablas hash en Java: funciones de hash, interfaz HashTable<K> y colisiones",
    "Hash (25 nov) 2 de 2 .pdf": "Implementaciones de tablas hash en Java: encadenamiento separado y direccionamiento abierto con rehashing",
    "ImplementaciónHash.pdf": "Implementación en C de una tabla hash para gestión de estructuras Persona con resolución de colisiones por lista enlazada",
    # Ficheros de montículos y AVL
    "5ÁrbolesAVL.pdf": "Árboles AVL: definición, propiedades de equilibrio, rotaciones simples y dobles, operaciones de inserción y eliminación, complejidad y detalles de implementación en Java",
    "2Izquierdistas.pdf": "Montículos izquierdistas ponderados: propiedad izquierdista y de orden de montículo, fusión eficiente, inserción, eliminación y complejidad, con implementación en Java",
}

document_list_description = "\n".join([f"- `{os.path.join('documentos_pdf', key)}`: {value}" for key, value in file_topic_mapping.items()])
source_selection_prompt_template = f"""
Eres un asistente experto en clasificar preguntas... (etc.)"""

def get_relevant_source_file(llm, user_query):
    prompt = PromptTemplate.from_template(source_selection_prompt_template)
    chain = prompt | llm
    response = chain.invoke({"user_query": user_query})
    content = response.content.strip().replace("`", "")
    if content in [os.path.join('documentos_pdf', key) for key in file_topic_mapping.keys()]:
        return content
    return None

# --- INICIO: DEFINICIÓN DE DOS PLANTILLAS DE PROMPT ---

# Plantilla 1: Se usará cuando SÍ encontremos contexto en el RAG
prompt_with_rag_template_str = """
Eres un tutor de programación experto y tu objetivo es personalizar la asistencia basándote en el historial del estudiante para fomentar la innovación y el pensamiento crítico. No debes dar respuestas directas. Tu método se basa en guiar al estudiante hacia la solución.

**Documentos de la Asignatura:**
{context}

**Contexto del Historial del Estudiante:**
{chat_history}

**Reglas Estrictas de Interacción:**
1. Usa el Método Socrático: nunca des la respuesta directa. Guía con preguntas.
2. Adapta la dificultad según el historial.
3. Fomenta la autoexplicación.
4. Da retroalimentación constructiva y personalizada.
5. Estimula la curiosidad: termina con preguntas abiertas.

**Implementación en Java o C según indique el estudiante.**


**Pregunta Actual del Estudiante:**
{question}

**Respuesta del Tutor:**"""

# Plantilla 2: Se usará como FALLBACK cuando el RAG no encuentre nada relevante
prompt_without_rag_template_str = """
Eres un tutor de programación experto y tu objetivo es personalizar la asistencia basándote en el historial del estudiante para fomentar la innovación y el pensamiento crítico. No debes dar respuestas directas. Tu método se basa en guiar al estudiante hacia la solución.

**Contexto del Historial del Estudiante:**
{chat_history}

**Reglas Estrictas de Interacción:**
1. Usa el Método Socrático: nunca des la respuesta directa. Guía con preguntas.
2. Adapta la dificultad según el historial.
3. Fomenta la autoexplicación.
4. Da retroalimentación constructiva y personalizada.
5. Estimula la curiosidad: termina con preguntas abiertas.

**Implementación en Java o C según indique el estudiante.**


**Pregunta Actual del Estudiante:**
{question}

**Respuesta del Tutor:**"""

# --- FIN: DEFINICIÓN DE DOS PLANTILLAS DE PROMPT ---


# --- INICIO DE LA APP ---
st.set_page_config(page_title="Tutor ED App", layout="wide")
st.header("🤖 Tutor de Estructuras de Datos (RAG Avanzado)")

# --- LOGIN ---
try:
    df_estudiantes = cargar_datos_estudiantes()
    idcv_value = st.query_params.get("idcv")
    nombre_value = st.query_params.get("nombre")
    if idcv_value and nombre_value:
        user_data = df_estudiantes[df_estudiantes['IDCV'] == str(idcv_value)]
        if not user_data.empty:
            st.session_state.authenticated = True
            st.session_state.user_idcv = idcv_value
            st.session_state.user_name = user_data.iloc[0]['Nombre']
        else:
            st.error(f"❌ Usuario no autorizado. IDCV recibido: {idcv_value}")
            st.stop()
    else:
        st.error("❌ Acceso no autorizado. Faltan los parámetros 'idcv' y 'nombre' en la URL.")
        st.stop()
except FileNotFoundError:
    st.error(f"Error crítico: El fichero de usuarios '{USERS_CSV_PATH_ENC}' no se encontró.")
    st.stop()

# --- INICIALIZACIÓN DE SERVICIOS ---
db = get_firestore_client()
api_key = st.secrets["GOOGLE_API_KEY"]
vectorstore = inicializar_vectorstore(api_key)
if vectorstore is None:
    st.stop()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key, temperature=0.5)

# --- INICIO DEL CHAT ---
st.title(f"Hola, {st.session_state.user_name}")

# --- Modificación para el botón de reseteo ---
with st.sidebar:
    st.header("Opciones de Chat")
    
    # El botón se habilita/deshabilita según el estado de la sesión
    if st.button("🗑️ Resetear Chat", 
                 help="Inicia una nueva conversación. Se desactivará hasta tu próximo mensaje.", 
                 disabled=st.session_state.get("reset_button_disabled", False)):
        reset_user_chat(db, st.session_state.user_idcv)
        st.session_state.reset_button_disabled = True # Desactivamos el botón inmediatamente
        st.rerun()

# Inicialización de variables de estado
if "messages" not in st.session_state:
    st.session_state.messages = load_active_user_history(db, st.session_state.user_idcv)
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "¿Sobre qué tema o estructura de datos tienes dudas hoy?"})
        
if "esperando_respuesta" not in st.session_state:
    st.session_state.esperando_respuesta = False
    
if "reset_button_disabled" not in st.session_state:
    st.session_state.reset_button_disabled = False

# Mostrar historial
for message in st.session_state.messages:
    if message.get("role") != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- LÓGICA PRINCIPAL DE CHAT ---
if prompt := st.chat_input("Escribe aquí tu duda...", disabled=st.session_state.esperando_respuesta):
    
    # Reactivar el botón de reseteo si el usuario interactúa
    if st.session_state.get("reset_button_disabled", False):
        st.session_state.reset_button_disabled = False

    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    save_message_to_history(db, st.session_state.user_idcv, user_message)
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.esperando_respuesta = True
    st.rerun()

if st.session_state.esperando_respuesta and st.session_state.messages[-1]["role"] == "user":
    try:
        with st.spinner("⏳ El tutor está pensando..."):
            current_prompt = st.session_state.messages[-1]["content"]
            
            # --- LÓGICA DE RAG ---
            source_file = get_relevant_source_file(llm, current_prompt)
            docs = []
            if source_file:
                candidate_docs = vectorstore.similarity_search(current_prompt, k=20)
                normalized_target_source = source_file.replace("\\", "/")
                filtered_docs = [doc for doc in candidate_docs if doc.metadata.get("source", "").replace("\\", "/") == normalized_target_source]
                docs = filtered_docs[:5]
            else:
                docs = vectorstore.similarity_search(current_prompt, k=5)
            
            # --- LÓGICA CONDICIONAL DE PROMPT ---
            last5 = st.session_state.messages[-6:-1] if len(st.session_state.messages) > 1 else []
            chat_hist = "\n\n".join([f"- {('Pregunta' if h['role'] == 'user' else 'Respuesta')}: {h['content']}" for h in last5]) or "El estudiante no tiene interacciones previas."

            if docs:
                st.success(f"✅ Se han encontrado {len(docs)} chunks relevantes. Usando modo RAG.")
                context = "\n\n".join([d.page_content for d in docs])
                template = PromptTemplate(template=prompt_with_rag_template_str, input_variables=["chat_history", "context", "question"])
                chain_input = {"chat_history": chat_hist, "context": context, "question": current_prompt}
            else:
                st.warning("⚠️ No se encontraron chunks relevantes. El tutor responderá desde su conocimiento general.")
                context = "No se encontró información en los documentos de la asignatura." 
                template = PromptTemplate(template=prompt_without_rag_template_str, input_variables=["chat_history", "question"])
                chain_input = {"chat_history": chat_hist, "question": current_prompt}
            
            with st.expander("🕵️‍♂️ **Ver Prompt Enviado al LLM**"):
                filled_prompt = template.format_prompt(**chain_input).to_string()
                st.text_area("Prompt Final Completo", filled_prompt, height=400)

            chain = template | llm
            resp_content = chain.invoke(chain_input).content

        with st.chat_message("assistant"):
            st.markdown(resp_content)
        assistant_message = {"role": "assistant", "content": resp_content}
        st.session_state.messages.append(assistant_message)
        save_message_to_history(db, st.session_state.user_idcv, assistant_message)

    except Exception as e:
        error_message = f"❌ Lo siento, ocurrió un error al procesar tu pregunta.\n\nDetalle: {str(e)}"
        st.error(error_message)
        st.code(traceback.format_exc(), language="python")
    
    finally:
        st.session_state.esperando_respuesta = False
        st.rerun()