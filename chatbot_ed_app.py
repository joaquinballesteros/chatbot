# == chatbot_ed_app.py ==
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

# --- CONSTANTES DE CIFRADO ---
fernet_key = st.secrets["encryption"]["key"].encode()
fernet = Fernet(fernet_key)

# --- RUTAS Y CONSTANTES ---
DATA_DIR = "data"
TMP_DIR = tempfile.gettempdir()  # Carpeta temporal del sistema

USERS_CSV_ENC = os.path.join(DATA_DIR, "users.csv.encrypted")
USERS_CSV_TMP = os.path.join(TMP_DIR, "users.csv.tmp")

HIST_ENC = os.path.join(DATA_DIR, "user_profiles.json.encrypted")
HIST_TMP = os.path.join(TMP_DIR, "user_profiles.json.tmp")

VECTORSTORE_ENC = os.path.join(DATA_DIR, ".RAG.encrypted")
VECTORSTORE_DIR = os.path.join(DATA_DIR, ".RAG")

# --- FUNCIONES DE CIFRADO/DESCIFRADO ---

def decrypt_file(src_path: str, dest_path: str):
    token = open(src_path, 'rb').read()
    data = fernet.decrypt(token)
    with open(dest_path, 'wb') as f:
        f.write(data)


def encrypt_file(src_path: str, dest_path: str):
    data = open(src_path, 'rb').read()
    token = fernet.encrypt(data)
    with open(dest_path, 'wb') as f:
        f.write(token)
    os.remove(src_path)

# --- CARGA DE USUARIOS ---
@st.cache_data(ttl=3600)
def cargar_datos_estudiantes():
    if not os.path.exists(USERS_CSV_TMP):
        decrypt_file(USERS_CSV_ENC, USERS_CSV_TMP)
    df = pd.read_csv(USERS_CSV_TMP, dtype=str)
    df['IDCV'] = df['IDCV'].str.strip()
    os.remove(USERS_CSV_TMP)  # Limpieza
    return df

# --- GESTIÓN DE HISTORIAL CIFRADO ---
def cargar_o_crear_historiales():
    if os.path.exists(HIST_ENC):
        decrypt_file(HIST_ENC, HIST_TMP)
    if not os.path.exists(HIST_TMP):
        with open(HIST_TMP, 'w', encoding='utf-8') as f:
            json.dump({}, f)

    try:
        with open(HIST_TMP, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ Error cargando historial: {e}")
        return {}


def guardar_historial(datos: dict):
    try:
        with open(HIST_TMP, 'w', encoding='utf-8') as f:
            json.dump(datos, f, ensure_ascii=False, indent=4)
        encrypt_file(HIST_TMP, HIST_ENC)
    except Exception as e:
        st.error("❌ Error al guardar el historial")
        st.code(traceback.format_exc(), language="python")



@st.cache_resource
def inicializar_vectorstore(api_key: str):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    # Vectorstore FAISS en memoria
    return FAISS.from_texts(["dummy"], embedding=embeddings)  # Puedes cargar documentos reales luego

# --- PLANTILLA PERSONALIZADA ---
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

# --- LOGIN ---
st.header("🤖 Tutor de Estructuras de Datos")

df_estudiantes = cargar_datos_estudiantes()

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


# --- CHAT ---
st.title(f"Hola, {st.session_state.user_name}")

# Inicialización del estado de la sesión
if "esperando_respuesta" not in st.session_state:
    st.session_state.esperando_respuesta = False
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¿Sobre qué tema o estructura de datos tienes dudas hoy?"}]

# Carga de perfiles e historial
user_profiles = cargar_o_crear_historiales()
uid = st.session_state.user_idcv
if uid not in user_profiles:
    user_profiles[uid] = {"nombre": st.session_state.user_name, "historial": []}
history = user_profiles[uid]["historial"]

# Inicializa LLM y vectorstore
api_key = st.secrets["GOOGLE_API_KEY"]
try:
    vectorstore = inicializar_vectorstore(api_key)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key, temperature=0.5)
except Exception as e:
    st.error("❌ Error al inicializar los servicios de IA:")
    st.code(traceback.format_exc(), language="python")
    st.stop()

# Mostrar mensajes previos del historial de la sesión
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- INICIO DE LA LÓGICA DE CHAT MEJORADA ---

# 1. Capturar la entrada del usuario con st.chat_input
#    Se deshabilita si st.session_state.esperando_respuesta es True.
if prompt := st.chat_input(
    "Escribe aquí tu duda...",
    disabled=st.session_state.esperando_respuesta
):
    # Añadir el mensaje del usuario al historial de la sesión y mostrarlo
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Activar el estado de "espera" y redibujar la pantalla para bloquear la entrada
    st.session_state.esperando_respuesta = True
    st.rerun()

# 2. Procesar la pregunta si estamos en estado de "espera"
#    Esta condición es clave: solo se ejecuta si estamos esperando respuesta Y el último mensaje fue del usuario.
#    Esto evita que se vuelva a generar una respuesta si el usuario simplemente recarga la página.
if st.session_state.esperando_respuesta and st.session_state.messages[-1]["role"] == "user":
    try:
        # Usar un spinner para dar feedback visual durante el procesamiento
        with st.spinner("⏳ El tutor está pensando..."):
            # Obtener la pregunta actual del historial de la sesión
            current_prompt = st.session_state.messages[-1]["content"]

            # Lógica RAG (igual que en tu código original)
            docs = vectorstore.similarity_search(current_prompt, k=5)
            context = "\n\n".join([d.page_content for d in docs])
            last5 = history[-5:]
            chat_hist = "\n".join([f"- {h['prompt']} => {h['respuesta']}" for h in last5]) or "El estudiante no tiene interacciones previas."

            template = PromptTemplate(
                template=prompt_template_str,
                input_variables=["chat_history", "context", "question"]
            )
            chain = template | llm
            
            # Usamos .invoke() que es la forma estándar y síncrona
            resp_content = chain.invoke({
                "chat_history": chat_hist,
                "context": context,
                "question": current_prompt
            }).content

        # Mostrar la respuesta del asistente
        with st.chat_message("assistant"):
            st.markdown(resp_content)

        # Guardar en el historial persistente y en el de la sesión
        history.append({"prompt": current_prompt, "respuesta": resp_content})
        guardar_historial(user_profiles)
        st.session_state.messages.append({"role": "assistant", "content": resp_content})

    except Exception as e:
        error_message = f"❌ Lo siento, ocurrió un error al procesar tu pregunta. Por favor, inténtalo de nuevo.\n\nDetalle: {str(e)}"
        st.error(error_message)
        st.code(traceback.format_exc(), language="python")
        st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    finally:
        # Desactivar el estado de "espera" y redibujar para habilitar la entrada
        st.session_state.esperando_respuesta = False
        st.rerun()

# --- SECCIÓN DE ADMINISTRADOR (Añadir al final del script) ---

# Cargar el ID de administrador desde los secretos
ADMIN_IDCV = st.secrets.get("app_config", {}).get("admin_idcv")

# Comprobar si el usuario actual es un administrador
if ADMIN_IDCV and st.session_state.user_idcv == ADMIN_IDCV:
    st.sidebar.title("Panel de Administración")
    st.sidebar.write("Descarga los datos para análisis.")

    # Asegurarse de que el fichero cifrado existe antes de ofrecer la descarga
    if os.path.exists(HIST_ENC):
        with open(HIST_ENC, "rb") as fp:
            st.sidebar.download_button(
                label="Descargar Historial Cifrado",
                data=fp,
                file_name="user_profiles.json.encrypted",
                mime="application/octet-stream"
            )