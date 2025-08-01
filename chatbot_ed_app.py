# == chatbot_ed_app.py (Versión con Firebase Firestore) ==
import os
import streamlit as st
import pandas as pd
from datetime import datetime

# --- LIBRERÍAS DE IA (LangChain y Google) ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import traceback

# --- LIBRERÍAS DE FIREBASE ---
import google.oauth2.service_account
from google.cloud import firestore

import asyncio

# --- RUTAS Y CONSTANTES (SIMPLIFICADO) ---
# Ya no necesitamos directorios de datos para el historial
TMP_DIR = os.path.join(os.getcwd(), "tmp") # Usamos un directorio temporal local si es necesario
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

USERS_CSV_PATH_ENC = os.path.join(os.getcwd(), "data", "users.csv.encrypted")
USERS_CSV_PATH_TMP = os.path.join(TMP_DIR, "users.csv.tmp")


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
    """
    Carga el historial de chat de un usuario desde Firestore,
    considerando solo los mensajes después del último reseteo.
    """
    historial_ref = db.collection('users').document(user_id).collection('historial')
    
    # 1. Buscar la marca del último reseteo
    reset_query = historial_ref.where("role", "==", "system").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1)
    reset_docs = list(reset_query.stream())
    
    last_reset_timestamp = None
    if reset_docs:
        last_reset_timestamp = reset_docs[0].get('timestamp')

    # 2. Obtener los mensajes posteriores a esa marca de tiempo
    if last_reset_timestamp:
        query = historial_ref.where("timestamp", ">", last_reset_timestamp).order_by("timestamp")
    else:
        # Si no hay reseteos, obtener todo el historial
        query = historial_ref.order_by("timestamp")
        
    docs = query.stream()
    
    # Convertir los documentos de Firestore a una lista de diccionarios
    messages = [doc.to_dict() for doc in docs]
    return messages

def save_message_to_history(db: firestore.Client, user_id: str, message: dict):
    """Guarda un único mensaje en el historial de Firestore del usuario."""
    # Añade una marca de tiempo del servidor para un ordenamiento fiable
    message_to_save = message.copy()
    message_to_save['timestamp'] = firestore.SERVER_TIMESTAMP
    
    # Añade el nuevo mensaje como un nuevo documento en la sub-colección
    db.collection('users').document(user_id).collection('historial').add(message_to_save)

def reset_user_chat(db: firestore.Client, user_id: str):
    """
    Registra un evento de reseteo en el historial de Firestore.
    No borra datos, solo añade una marca.
    """
    reset_message = {
        "role": "system",
        "content": f"--- CHAT RESETEADO POR EL USUARIO ---",
        # El timestamp se añadirá en save_message_to_history
    }
    save_message_to_history(db, user_id, reset_message)
    # Limpia el estado de la sesión actual para reflejar el reseteo en la UI
    st.session_state.messages = []


# --- FUNCIONES AUXILIARES (CIFRADO SOLO PARA EL CSV DE USUARIOS) ---
@st.cache_data(ttl=3600)
def cargar_datos_estudiantes():
    """Descifra y carga el CSV de estudiantes."""
    # Asumimos que la clave de cifrado está en st.secrets
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
# Reemplaza tu función actual con esta versión

@st.cache_resource
def inicializar_vectorstore(api_key: str):
    """
    Inicializa el vectorstore de FAISS con los embeddings de Google,
    asegurando que exista un bucle de eventos asyncio.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:  # 'RuntimeError: There is no current event loop...'
        asyncio.set_event_loop(asyncio.new_event_loop())
    # --- FIN DE LA CORRECCIÓN ---

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    # Vectorstore FAISS en memoria
    return FAISS.from_texts(["dummy"], embedding=embeddings)

# --- PLANTILLA DE PROMPT (SIN CAMBIOS) ---
prompt_template_str = """
Eres un tutor de programación experto... (etc.)
"""

# --- CONFIGURACIÓN E INICIO DE LA APP ---
st.set_page_config(page_title="Tutor ED App", layout="wide")
st.header("🤖 Tutor de Estructuras de Datos v2 (con Firestore)")

# --- LOGIN (SIN CAMBIOS) ---
df_estudiantes = cargar_datos_estudiantes()
# ... (el resto de tu lógica de login con st.query_params va aquí) ...
# Obtener parámetros de la URL
params = st.query_params
idcv_param = st.query_params.get("idcv", [None])
nombre_param = st.query_params.get("nombre", [None])

# Extraer el valor real de los parámetros (primer elemento de la lista)
idcv_value = idcv_param[0] if isinstance(idcv_param, list) else idcv_param
nombre_value = nombre_param[0] if isinstance(nombre_param, list) else nombre_param

# Validación: solo si vienen por GET y son válidos
if idcv_value and nombre_value:
    user = df_estudiantes[df_estudiantes['IDCV'] == idcv_value]
    if not user.empty:
        st.session_state.authenticated = True
        st.session_state.user_idcv = idcv_value
        st.session_state.user_name = user.iloc[0]['Nombre']
    else:
        st.error(f"❌ Este usuario no tiene permiso para usar el tutor.\n\nPor favor, contacta con el profesor de la asignatura. \nIDCV recibido: {idcv_value}\nNombre recibido: {nombre_value}")
        st.stop()
else:
    st.error(f"❌ Acceso no autorizado. Faltan credenciales válidas en la URL.\n\nPor favor, contacta con el profesor de la asignatura.")
    st.stop()


# --- INICIALIZACIÓN DE SERVICIOS ---
db = get_firestore_client()
api_key = st.secrets["GOOGLE_API_KEY"]
vectorstore = inicializar_vectorstore(api_key)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key, temperature=0.5)

# --- INICIO DEL CHAT ---
st.title(f"Hola, {st.session_state.user_name}")

# Barra lateral con opciones
with st.sidebar:
    st.header("Opciones de Chat")
    if st.button("🗑️ Resetear Chat", help="Inicia una nueva conversación."):
        reset_user_chat(db, st.session_state.user_idcv)
        st.rerun()

# Inicialización del historial de chat en la sesión
if "messages" not in st.session_state:
    st.session_state.messages = load_active_user_history(db, st.session_state.user_idcv)
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "¿Sobre qué tema o estructura de datos tienes dudas hoy?"})
if "esperando_respuesta" not in st.session_state:
    st.session_state.esperando_respuesta = False

# Mostrar mensajes del historial activo
for message in st.session_state.messages:
    if message.get("role") != "system": # No mostrar mensajes de sistema
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- LÓGICA DE CHAT CON ESTADO DE ESPERA ---
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

            # Obtener el historial activo para el contexto del LLM
            history_for_prompt = st.session_state.messages
            
            docs = vectorstore.similarity_search(current_prompt, k=5)
            context = "\n\n".join([d.page_content for d in docs])
            
            # Usar los últimos 5 mensajes de la sesión activa como contexto
            last5 = history_for_prompt[-6:-1] if len(history_for_prompt) > 1 else []
            chat_hist = "\n".join([f"- {('Pregunta' if h['role'] == 'user' else 'Respuesta')}: {h['content']}" for h in last5]) or "El estudiante no tiene interacciones previas."

            template = PromptTemplate(template=prompt_template_str, input_variables=["chat_history", "context", "question"])
            chain = template | llm
            
            resp_content = chain.invoke({
                "chat_history": chat_hist,
                "context": context,
                "question": current_prompt
            }).content

        with st.chat_message("assistant"):
            st.markdown(resp_content)

        assistant_message = {"role": "assistant", "content": resp_content}
        st.session_state.messages.append(assistant_message)
        save_message_to_history(db, st.session_state.user_idcv, assistant_message)

    except Exception as e:
        error_message = f"❌ Lo siento, ocurrió un error al procesar tu pregunta.\n\nDetalle: {str(e)}"
        st.error(error_message)
        st.code(traceback.format_exc(), language="python")
        # Guardar también el mensaje de error en el historial
        error_msg_to_save = {"role": "assistant", "content": error_message}
        st.session_state.messages.append(error_msg_to_save)
        save_message_to_history(db, st.session_state.user_idcv, error_msg_to_save)
    
    finally:
        st.session_state.esperando_respuesta = False
        st.rerun()