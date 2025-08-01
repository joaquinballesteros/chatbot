# == chatbot_ed_app.py (Versión Refactorizada) ==
import os
import shutil
import tempfile
import json
import streamlit as st
from cryptography.fernet import Fernet
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from create_rag import decrypt_folder
from langchain_core.documents import Document
import asyncio
import traceback
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
import datetime

# --- CONSTANTES DE CIFRADO ---
fernet_key = st.secrets["encryption"]["key"].encode()
fernet = Fernet(fernet_key)

# --- RUTAS Y CONSTANTES ---
DATA_DIR = "data"
TMP_DIR = tempfile.gettempdir()

USERS_CSV_ENC = os.path.join(DATA_DIR, "users.csv.encrypted")
USERS_CSV_TMP = os.path.join(TMP_DIR, "users.csv.tmp")

# MODIFICADO: Ya no usamos un único fichero de historial. Se ha eliminado HIST_ENC y HIST_TMP.

# --- SINCRONIZACIÓN CON GOOGLE DRIVE ---
# --- SINCRONIZACIÓN CON GOOGLE DRIVE (CON DEPURACIÓN) ---
def sincronizar_con_drive(fichero_local: str, nombre_fichero_drive: str):
    """Sube o actualiza un fichero en Google Drive con mensajes de depuración."""
    # DEBUG: Mensaje al iniciar la función
    st.toast(f"▶️ Iniciando sincronización para: {nombre_fichero_drive}")
    try:
        # DEBUG: Confirmar que las credenciales se cargan
        st.info("🕵️‍♂️ DEBUG: Cargando credenciales de GCP desde st.secrets...")
        creds_dict = st.secrets["gcp_service_account"]
        creds = google.oauth2.service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        st.info("✅ DEBUG: Credenciales cargadas. Construyendo servicio de Drive...")
        
        service = build('drive', 'v3', credentials=creds)
        folder_id = st.secrets["gdrive"]["folder_id"]
        st.info(f"✅ DEBUG: Servicio de Drive construido. Buscando en Folder ID: ...{folder_id[-10:]}")

        # DEBUG: Buscar si el fichero ya existe
        query = f"name='{nombre_fichero_drive}' and '{folder_id}' in parents and trashed=false"
        st.info(f"🕵️‍♂️ DEBUG: Ejecutando query en Drive: {query}")
        response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        files = response.get('files', [])
        st.info(f"✅ DEBUG: Búsqueda en Drive completada. Ficheros encontrados: {files}")

        media = MediaFileUpload(fichero_local, mimetype='application/octet-stream', resumable=True)

        if not files:
            # DEBUG: Crear fichero si no existe
            st.warning(f"⚠️ DEBUG: Fichero no encontrado en Drive. Intentando crear '{nombre_fichero_drive}'...")
            file_metadata = {'name': nombre_fichero_drive, 'parents': [folder_id]}
            service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            st.success(f"✅ ¡ÉXITO! Fichero creado en Drive.")
        else:
            # DEBUG: Actualizar fichero si existe
            file_id = files[0].get('id')
            st.warning(f"⚠️ DEBUG: Fichero encontrado con ID: {file_id}. Intentando actualizar...")
            service.files().update(fileId=file_id, media_body=media).execute()
            st.success(f"✅ ¡ÉXITO! Fichero actualizado en Drive.")

    except HttpError as error:
        # DEBUG: Captura de error de API específico
        st.error(f"❌ ERROR DE API DE GOOGLE al sincronizar con Drive.")
        st.exception(error)
    except Exception as e:
        # DEBUG: Captura de cualquier otro error
        st.error(f"❌ ERROR INESPERADO al sincronizar con Drive.")
        st.exception(e) # st.exception muestra el error completo y el traceback

# --- GESTIÓN DE HISTORIAL INDIVIDUAL (CON DEPURACIÓN) ---

def save_user_history(user_id: str, new_message: dict):
    """Añade un mensaje al historial y lo guarda con mensajes de depuración."""
    # DEBUG: Mensaje al iniciar el guardado
    st.toast(f"▶️ Guardando mensaje para usuario: {user_id}")
    enc_path, tmp_path = get_user_filepaths(user_id) # Función sin cambios
    
    try:
        full_history = []
        if os.path.exists(enc_path):
            decrypt_file(enc_path, tmp_path)
            try:
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    full_history = json.load(f)
            except json.JSONDecodeError:
                full_history = []
        
        # DEBUG: Confirmar que el historial se cargó y se añadió el mensaje
        st.info(f"🕵️‍♂️ DEBUG: Historial cargado con {len(full_history)} mensajes. Añadiendo nuevo mensaje.")
        full_history.append(new_message)
        st.info(f"✅ DEBUG: Mensaje añadido. Total ahora: {len(full_history)}.")
        
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(full_history, f, ensure_ascii=False, indent=4)
        
        encrypt_file(tmp_path, enc_path)
        # DEBUG: Confirmar que el fichero local se ha creado/actualizado
        st.success(f"✅ DEBUG: Fichero local cifrado guardado en: {enc_path}")
        
        # Llamada a la sincronización
        sincronizar_con_drive(enc_path, os.path.basename(enc_path))
        
    except Exception as e:
        st.error("❌ ERROR INESPERADO durante save_user_history.")
        st.exception(e)


# --- FUNCIONES DE CIFRADO/DESCIFRADO ---
def decrypt_file(src_path: str, dest_path: str):
    if not os.path.exists(src_path): return
    with open(src_path, 'rb') as f:
        token = f.read()
    data = fernet.decrypt(token)
    with open(dest_path, 'wb') as f:
        f.write(data)

def encrypt_file(src_path: str, dest_path: str):
    with open(src_path, 'rb') as f:
        data = f.read()
    token = fernet.encrypt(data)
    with open(dest_path, 'wb') as f:
        f.write(token)
    if os.path.exists(src_path): os.remove(src_path)

# --- CARGA DE USUARIOS ---
@st.cache_data(ttl=3600)
def cargar_datos_estudiantes():
    if not os.path.exists(USERS_CSV_TMP):
        decrypt_file(USERS_CSV_ENC, USERS_CSV_TMP)
    df = pd.read_csv(USERS_CSV_TMP, dtype=str)
    df['IDCV'] = df['IDCV'].str.strip()
    if os.path.exists(USERS_CSV_TMP): os.remove(USERS_CSV_TMP)
    return df

# --- NUEVO: GESTIÓN DE HISTORIAL INDIVIDUAL ---

def get_user_filepaths(user_id: str):
    """Devuelve las rutas de los ficheros cifrado y temporal para un usuario."""
    safe_id = "".join(c for c in user_id if c.isalnum())
    filename_base = f"history_{safe_id}.json"
    enc_path = os.path.join(DATA_DIR, f"{filename_base}.encrypted")
    tmp_path = os.path.join(TMP_DIR, filename_base)
    return enc_path, tmp_path

def load_active_user_history(user_id: str):
    """
    Carga el historial completo de un usuario, pero solo devuelve la parte
    activa (mensajes después del último reseteo).
    """
    enc_path, tmp_path = get_user_filepaths(user_id)
    if not os.path.exists(enc_path):
        return []

    try:
        decrypt_file(enc_path, tmp_path)
        with open(tmp_path, 'r', encoding='utf-8') as f:
            full_history = json.load(f)
        if os.path.exists(tmp_path): os.remove(tmp_path)

        # Buscar el índice del último reseteo
        last_reset_index = -1
        for i, msg in enumerate(reversed(full_history)):
            if msg.get("role") == "system" and "CHAT RESETEADO" in msg.get("content", ""):
                last_reset_index = len(full_history) - 1 - i
                break
        
        # Devolver solo la parte activa del historial
        return full_history[last_reset_index + 1:] if last_reset_index != -1 else full_history

    except (json.JSONDecodeError, FileNotFoundError):
        return []


def reset_user_chat(user_id: str):
    """Añade una marca de reseteo al historial del usuario."""
    reset_message = {
        "role": "system",
        "content": f"--- CHAT RESETEADO POR EL USUARIO A LAS {datetime.datetime.now().isoformat()} ---"
    }
    save_user_history(user_id, reset_message)
    # Limpia el estado de la sesión actual para reflejar el reseteo en la UI
    st.session_state.messages = []


@st.cache_resource
def inicializar_vectorstore(api_key: str):
    try: asyncio.get_running_loop()
    except RuntimeError: asyncio.set_event_loop(asyncio.new_event_loop())
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    return FAISS.from_texts(["dummy"], embedding=embeddings)

# --- PLANTILLA (Sin cambios) ---
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

# --- CONFIGURACIÓN STREAMLIT ---
st.set_page_config(page_title="Tutor ED App", layout="wide")

# --- LOGIN (Sin cambios) ---
st.header("🤖 Tutor de Estructuras de Datos")
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
        st.error(f"❌ Este usuario no tiene permiso para usar el tutor. IDCV: {idcv_value}, Nombre: {nombre_value}")
        st.stop()
else:
    st.error("❌ Acceso no autorizado. Faltan credenciales válidas en la URL.")
    st.stop()

# --- INICIO DEL CHAT ---
st.title(f"Hola, {st.session_state.user_name}")

# MODIFICADO: Barra lateral con botón de reseteo para el usuario
with st.sidebar:
    st.header("Opciones de Chat")
    if st.button("🗑️ Resetear Chat", help="Inicia una nueva conversación. Tu historial anterior se guardará pero no se usará como contexto."):
        reset_user_chat(st.session_state.user_idcv)
        st.rerun() # Recarga la app para mostrar el chat vacío

# Inicialización del estado de la sesión
if "esperando_respuesta" not in st.session_state:
    st.session_state.esperando_respuesta = False
if "messages" not in st.session_state:
    # MODIFICADO: Carga el historial activo del usuario específico
    st.session_state.messages = load_active_user_history(st.session_state.user_idcv)
    # Añadir mensaje de bienvenida si el historial está vacío
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "¿Sobre qué tema o estructura de datos tienes dudas hoy?"})


# Inicializa LLM y vectorstore
api_key = st.secrets["GOOGLE_API_KEY"]
try:
    vectorstore = inicializar_vectorstore(api_key)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key, temperature=0.5)
except Exception as e:
    st.error(f"❌ Error al inicializar los servicios de IA: {e}")
    st.stop()

# Mostrar mensajes previos del historial de la sesión
for message in st.session_state.messages:
    # No mostrar los mensajes del sistema (como los de reseteo)
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Lógica de chat con estado de espera (sin cambios funcionales, solo en el guardado)
if prompt := st.chat_input("Escribe aquí tu duda...", disabled=st.session_state.esperando_respuesta):
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    save_user_history(st.session_state.user_idcv, user_message) # Guardar inmediatamente
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.esperando_respuesta = True
    st.rerun()

if st.session_state.esperando_respuesta and st.session_state.messages[-1]["role"] == "user":
    try:
        with st.spinner("⏳ El tutor está pensando..."):
            current_prompt = st.session_state.messages[-1]["content"]

            # MODIFICADO: Carga el historial activo para pasarlo al LLM
            history_for_prompt = load_active_user_history(st.session_state.user_idcv)
            
            docs = vectorstore.similarity_search(current_prompt, k=5)
            context = "\n\n".join([d.page_content for d in docs])
            last5 = history_for_prompt[-6:-1] # Usar los 5 mensajes previos al actual
            chat_hist = "\n".join([f"- Pregunta: {h['content']}" if h['role'] == 'user' else f"- Respuesta: {h['content']}" for h in last5]) or "El estudiante no tiene interacciones previas."

            template = PromptTemplate(template=prompt_template_str, input_variables=["chat_history", "context", "question"])
            chain = template | llm





            # --- INICIO DE BLOQUE DE DEPURACIÓN ---
            with st.expander("🕵️‍♂️ Ver datos enviados al modelo"):
                st.info("Contexto extraído de los documentos (RAG):")
                st.text(context)
                st.info("Historial de chat usado como contexto:")
                st.text(chat_hist)
                st.info("Pregunta actual:")
                st.text(current_prompt)
            # --- FIN DE BLOQUE DE DEPURACIÓN ---





            
            resp_content = chain.invoke({
                "chat_history": chat_hist,
                "context": context,
                "question": current_prompt
            }).content

        with st.chat_message("assistant"):
            st.markdown(resp_content)

        assistant_message = {"role": "assistant", "content": resp_content}
        st.session_state.messages.append(assistant_message)
        save_user_history(st.session_state.user_idcv, assistant_message) # Guardar respuesta

    except Exception as e:
        error_message = f"❌ Lo siento, ocurrió un error. Por favor, inténtalo de nuevo.\n\nDetalle: {str(e)}"
        st.error(error_message)
        assistant_message = {"role": "assistant", "content": error_message}
        st.session_state.messages.append(assistant_message)
        save_user_history(st.session_state.user_idcv, assistant_message)
    
    finally:
        st.session_state.esperando_respuesta = False
        st.rerun()

# --- ELIMINADO: SECCIÓN DE ADMINISTRADOR ---
# Todo el bloque de código del panel de administrador, incluido el de depuración, ha sido eliminado.