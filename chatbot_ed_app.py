# == chatbot_ed_app.py (Versión Final con Firebase y RAG Persistente) ==
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
USERS_CSV_PATH_ENC = os.path.join("data", "users.csv.encrypted")
USERS_CSV_PATH_TMP = os.path.join(TMP_DIR, "users.csv.tmp")

# Crear directorio temporal si no existe
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)


# --- CONEXIÓN A FIREBASE (CACHEADA PARA EFICIENCIA) ---
@st.cache_resource
def get_firestore_client():
    """Se conecta a Firestore usando las credenciales de Streamlit Secrets."""
    key_dict = st.secrets["firebase_service_account"]
    creds = google.oauth2.service_account.Credentials.from_service_account_info(key_dict)
    db = firestore.Client(credentials=creds)
    return db


# --- GESTIÓN DE HISTORIAL CON FIRESTORE ---
def load_active_user_history(db: firestore.Client, user_id: str):
    """Carga el historial desde Firestore después del último reseteo."""
    historial_ref = db.collection('users').document(user_id).collection('historial')
    reset_query = historial_ref.where("role", "==", "system").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1)
    reset_docs = list(reset_query.stream())
    
    last_reset_timestamp = None
    if reset_docs:
        last_reset_timestamp = reset_docs[0].get('timestamp')

    if last_reset_timestamp:
        query = historial_ref.where("timestamp", ">", last_reset_timestamp).order_by("timestamp")
    else:
        query = historial_ref.order_by("timestamp")
        
    docs = query.stream()
    return [doc.to_dict() for doc in docs]

def save_message_to_history(db: firestore.Client, user_id: str, message: dict):
    """Guarda un mensaje en el historial de Firestore del usuario."""
    message_to_save = message.copy()
    message_to_save['timestamp'] = firestore.SERVER_TIMESTAMP
    db.collection('users').document(user_id).collection('historial').add(message_to_save)

def reset_user_chat(db: firestore.Client, user_id: str):
    """Registra un evento de reseteo en el historial."""
    reset_message = {"role": "system", "content": f"--- CHAT RESETEADO POR EL USUARIO ---"}
    save_message_to_history(db, user_id, reset_message)
    st.session_state.messages = []


# --- FUNCIONES AUXILIARES ---
@st.cache_data(ttl=3600)
def cargar_datos_estudiantes():
    """Descifra y carga el CSV de estudiantes."""
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
    """
    Carga el índice FAISS pre-construido desde el disco. Si no existe, muestra un error.
    """
    if not os.path.exists(INDEX_PATH):
        st.error("El índice vectorial (faiss_index) no se encontró en el repositorio.")
        st.warning("Por favor, genera el índice localmente con 'create_and_save_index.py' y súbelo a GitHub.")
        return None

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    vectorstore = FAISS.load_local(
        INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    st.sidebar.success("✅ Índice RAG cargado.", icon="📚")
    return vectorstore

# --- PLANTILLA DE PROMPT ---
prompt_template_str = """
Eres un tutor de programación experto y tu objetivo es personalizar la asistencia basándote en el historial del estudiante para fomentar la innovación y el pensamiento crítico. No debes dar respuestas directas. Tu método se basa en guiar al estudiante hacia la solución.

**Contexto del Historial del Estudiante:**
{chat_history}

**Documentos de la Asignatura:**
{context}

**Reglas Estrictas de Interacción:**
1. Usa el Método Socrático: nunca des la respuesta directa. Guía con preguntas.
2. Adapta la dificultad según el historial.
3. Fomenta la autoexplicación.
4. Da retroalimentación constructiva y personalizada.
5. Estimula la curiosidad: termina con preguntas abiertas.

**Implementación en Java o C según indique el estudiante.**

**Pregunta Actual:**
{question}

**Respuesta del Tutor:**"""

# --- CONFIGURACIÓN E INICIO DE LA APP ---
st.set_page_config(page_title="Tutor ED App", layout="wide")
st.header("🤖 Tutor de Estructuras de Datos")

# --- LOGIN ---
df_estudiantes = cargar_datos_estudiantes()
params = st.query_params
idcv_param = st.query_params.get("idcv", [None])
nombre_param = st.query_params.get("nombre", [None])
idcv_value = idcv_param[0] if isinstance(idcv_param, list) else idcv_param
nombre_value = nombre_param[0] if isinstance(nombre_param, list) else nombre_param

if idcv_value and nombre_value:
    user = df_estudiantes[df_estudiantes['IDCV'] == idcv_value]
    if not user.empty:
        st.session_state.authenticated = True
        st.session_state.user_idcv = idcv_value
        st.session_state.user_name = user.iloc[0]['Nombre']
    else:
        st.error(f"❌ Usuario no autorizado. IDCV: {idcv_value}")
        st.stop()
else:
    st.error(f"❌ Acceso no autorizado. Faltan credenciales.")
    st.stop()

# --- INICIALIZACIÓN DE SERVICIOS ---
db = get_firestore_client()
api_key = st.secrets["GOOGLE_API_KEY"]
vectorstore = inicializar_vectorstore(api_key)

# Detiene la ejecución si el índice vectorial no está cargado
if vectorstore is None:
    st.stop()
    
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key, temperature=0.5)

# --- INICIO DEL CHAT ---
st.title(f"Hola, {st.session_state.user_name}")

with st.sidebar:
    st.header("Opciones de Chat")
    if st.button("🗑️ Resetear Chat", help="Inicia una nueva conversación."):
        reset_user_chat(db, st.session_state.user_idcv)
        st.rerun()

# Inicialización de la sesión de chat
if "messages" not in st.session_state:
    st.session_state.messages = load_active_user_history(db, st.session_state.user_idcv)
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "¿Sobre qué tema o estructura de datos tienes dudas hoy?"})
if "esperando_respuesta" not in st.session_state:
    st.session_state.esperando_respuesta = False

# Mostrar historial de chat
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
            
            # Búsqueda de contexto en el RAG
            docs = vectorstore.similarity_search(current_prompt, k=5)
            context = "\n\n".join([d.page_content for d in docs])
            
            # Preparación del historial para el prompt
            last5 = st.session_state.messages[-6:-1] if len(st.session_state.messages) > 1 else []
            chat_hist = "\n".join([f"- {('Pregunta' if h['role'] == 'user' else 'Respuesta')}: {h['content']}" for h in last5]) or "El estudiante no tiene interacciones previas."

            # Creación de la plantilla
            template = PromptTemplate(template=prompt_template_str, input_variables=["chat_history", "context", "question"])
            
            # Expansor de depuración para ver el prompt final
            with st.expander("🕵️‍♂️ **Ver Prompt Enviado al LLM**"):
                # La forma correcta de "renderizar" el prompt para visualizarlo
                filled_prompt = template.format_prompt(
                    chat_history=chat_hist, 
                    context=context, 
                    question=current_prompt
                ).to_string()
                st.text_area("Prompt Final Completo (tal como lo ve el LLM)", filled_prompt, height=400)

            # Creación e invocación de la cadena de LangChain
            chain = template | llm
            resp_content = chain.invoke({
                "chat_history": chat_hist,
                "context": context,
                "question": current_prompt
            }).content

        # Mostrar y guardar respuesta
        with st.chat_message("assistant"):
            st.markdown(resp_content)
        assistant_message = {"role": "assistant", "content": resp_content}
        st.session_state.messages.append(assistant_message)
        save_message_to_history(db, st.session_state.user_idcv, assistant_message)

    except Exception as e:
        error_message = f"❌ Lo siento, ocurrió un error al procesar tu pregunta.\n\nDetalle: {str(e)}"
        st.error(error_message)
        st.code(traceback.format_exc(), language="python")
        error_msg_to_save = {"role": "assistant", "content": error_message}
        st.session_state.messages.append(error_msg_to_save)
        save_message_to_history(db, st.session_state.user_idcv, error_msg_to_save)
    
    finally:
        st.session_state.esperando_respuesta = False
        st.rerun()