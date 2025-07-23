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

if "esperando_respuesta" not in st.session_state:
    st.session_state.esperando_respuesta = False

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
user_profiles = cargar_o_crear_historiales()
uid = st.session_state.user_idcv
if uid not in user_profiles:
    user_profiles[uid]={"nombre":st.session_state.user_name,"historial":[]}
history=user_profiles[uid]["historial"]

# Inicializa LLM y vectorstore
api_key=st.secrets["GOOGLE_API_KEY"]
try:
    vectorstore = inicializar_vectorstore(api_key)
except Exception as e:
    st.error("❌ Error al inicializar el vectorstore:")
    st.code(traceback.format_exc(), language="python")
    st.stop()

llm=ChatGoogleGenerativeAI(model="gemini-2.5-pro",google_api_key=api_key,temperature=0.5)

# Mensajes previos\if "messages" not in st.session_state:
st.session_state.messages=[{"role":"assistant","content":"¿Sobre qué tema o estructura de datos tienes dudas hoy?"}]
for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

# Entrada usuario
if st.session_state.esperando_respuesta:
    st.chat_input("⏳ Esperando respuesta...", disabled=True)
else:
    with st.form("chat_form", clear_on_submit=True):
        prompt = st.text_input("Escribe aquí tu duda...", disabled=st.session_state.esperando_respuesta, placeholder="⏳ Esperando respuesta del tutor..." if st.session_state.esperando_respuesta else "")
        submitted = st.form_submit_button("Enviar")

    if submitted and prompt and not st.session_state.esperando_respuesta:
        st.session_state.esperando_respuesta = True
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        try:
            docs = vectorstore.similarity_search(prompt, k=5)
            context = "\n\n".join([d.page_content for d in docs])
            last5 = history[-5:]
            chat_hist = "\n".join([f"- {h['prompt']} => {h['respuesta']}" for h in last5]) or "El estudiante no tiene interacciones previas."

            template = PromptTemplate(template=prompt_template_str, input_variables=["chat_history", "context", "question"])
            chain = LLMChain(llm=llm, prompt=template)
            resp = chain.invoke({"chat_history": chat_hist, "context": context, "question": prompt}).get("text")

            st.chat_message("assistant").markdown(resp)
            history.append({"prompt": prompt, "respuesta": resp})
            guardar_historial(user_profiles)
            st.session_state.messages.append({"role": "assistant", "content": resp})
        except Exception as e:
            st.error("❌ Error procesando la respuesta:")
            st.code(traceback.format_exc(), language="python")
        finally:
            st.session_state.esperando_respuesta = False
