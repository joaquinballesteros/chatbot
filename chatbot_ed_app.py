# == chatbot_ed_app.py (Versión Final v4 - RAG "Retrieve-then-Filter" y Gemini 1.5 Pro) ==
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
    from cryptography.fernet import Fernet
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
    "tema1_introduccion.pdf": "Introducción a algoritmos y estructuras de datos",
    "tema2_pilas_y_colas.pdf": "Pilas (Stacks) y Colas (Queues) en Java",
    "tema3_arboles.pdf": "Árboles binarios, BST y árboles AVL",
    "tema4_implementacion_C.pdf": "Implementación de estructuras de datos en el lenguaje C"
}
document_list_description = "\n".join([f"- `{os.path.join('documentos_pdf', key)}`: {value}" for key, value in file_topic_mapping.items()])
source_selection_prompt_template = f"""
Eres un asistente experto en clasificar preguntas. Tu tarea es determinar cuál de los siguientes documentos es el más relevante para responder a la pregunta del usuario.

Aquí tienes la lista de documentos disponibles y de qué trata cada uno:
{document_list_description}

Pregunta del usuario: "{{user_query}}"

Analiza la pregunta y responde ÚNICA Y EXCLUSIVAMENTE con la ruta del fichero más relevante de la lista. Si ningún fichero parece directamente relevante, responde con "NONE".
Respuesta:
"""

def get_relevant_source_file(llm, user_query):
    prompt = PromptTemplate.from_template(source_selection_prompt_template)
    chain = prompt | llm
    response = chain.invoke({"user_query": user_query})
    content = response.content.strip().replace("`", "")
    if content in [os.path.join('documentos_pdf', key) for key in file_topic_mapping.keys()]:
        return content
    return None

# --- PLANTILLA DE PROMPT PRINCIPAL ---
prompt_template_str = """
Eres un tutor de programación experto... (etc.)"""

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
# ... (código del sidebar y de inicialización de la sesión de chat sin cambios) ...
with st.sidebar:
    st.header("Opciones de Chat")
    if st.button("🗑️ Resetear Chat", help="Inicia una nueva conversación."):
        reset_user_chat(db, st.session_state.user_idcv)
        st.rerun()
if "messages" not in st.session_state:
    st.session_state.messages = load_active_user_history(db, st.session_state.user_idcv)
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "¿Sobre qué tema o estructura de datos tienes dudas hoy?"})
if "esperando_respuesta" not in st.session_state:
    st.session_state.esperando_respuesta = False
for message in st.session_state.messages:
    if message.get("role") != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- LÓGICA PRINCIPAL DE CHAT ---
if prompt := st.chat_input("Escribe aquí tu duda...", disabled=st.session_state.esperando_respuesta):
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
            
            # --- INICIO DEL RAG "RETRIEVE-THEN-FILTER" ---
            st.info("Paso 1: Planificando qué documento consultar...")
            source_file = get_relevant_source_file(llm, current_prompt)
            
            docs = []
            if source_file:
                st.info(f"✅ Decisión: La búsqueda se centrará en el fichero `{source_file}`.")
                st.info("Paso 2: Recuperando un lote amplio de candidatos...")
                # Recuperamos más documentos de los necesarios para tener dónde elegir
                candidate_docs = vectorstore.similarity_search(current_prompt, k=20)
                
                st.info("Paso 3: Filtrando los candidatos manualmente...")
                # Filtramos la lista en Python
                filtered_docs = [doc for doc in candidate_docs if doc.metadata.get("source") == source_file]
                
                # Nos quedamos con los 5 mejores de la lista ya filtrada
                docs = filtered_docs[:5]
                st.success(f"✅ Se han encontrado {len(docs)} chunks relevantes en el documento correcto.")
            else:
                st.warning("⚠️ No se ha podido determinar un fichero específico. Se realizará una búsqueda general.")
                docs = vectorstore.similarity_search(current_prompt, k=5)
            # --- FIN DEL RAG ---

            context = "\n\n".join([d.page_content for d in docs])
            
            last5 = st.session_state.messages[-6:-1] if len(st.session_state.messages) > 1 else []
            chat_hist = "\n\n".join([f"- {('Pregunta' if h['role'] == 'user' else 'Respuesta')}: {h['content']}" for h in last5]) or "El estudiante no tiene interacciones previas."

            template = PromptTemplate(template=prompt_template_str, input_variables=["chat_history", "context", "question"])
            
            with st.expander("🕵️‍♂️ **Ver Prompt Enviado al LLM**"):
                filled_prompt = template.format_prompt(chat_history=chat_hist, context=context, question=current_prompt).to_string()
                st.text_area("Prompt Final Completo", filled_prompt, height=400)

            chain = template | llm
            resp_content = chain.invoke({"chat_history": chat_hist, "context": context, "question": current_prompt}).content

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