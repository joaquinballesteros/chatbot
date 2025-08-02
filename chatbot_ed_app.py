
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
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import StructuredQueryOutputParser, get_query_constructor_prompt

# --- LIBRERÍAS DE FIREBASE ---
import google.oauth2.service_account
from google.cloud import firestore

# --- RUTAS Y CONSTANTES (CORREGIDAS Y SIMPLIFICADAS) ---
TMP_DIR = "tmp"
INDEX_PATH = "faiss_index"
USERS_CSV_PATH_ENC = "data/users.csv.encrypted" # Ruta relativa directa
USERS_CSV_PATH_TMP = os.path.join(TMP_DIR, "users.csv.tmp")

if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

# --- CONEXIÓN A FIREBASE (CACHEADA) ---
@st.cache_resource
def get_firestore_client():
    key_dict = st.secrets["firebase_service_account"]
    creds = google.oauth2.service_account.Credentials.from_service_account_info(key_dict)
    db = firestore.Client(credentials=creds)
    return db

# --- GESTIÓN DE HISTORIAL CON FIRESTORE ---
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
def inicializar_vectorstore_and_retriever(_llm, api_key: str):
    if not os.path.exists(INDEX_PATH):
        st.error(f"Índice vectorial '{INDEX_PATH}' no encontrado.")
        st.warning("Genera el índice localmente y súbelo a GitHub.")
        return None

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    metadata_field_info = [
        AttributeInfo(name="source", description="El fichero PDF de origen del documento.", type="string"),
        AttributeInfo(name="topic", description="El tema principal del contenido del documento. Ej: 'Pilas y Colas en Java'.", type="string"),
    ]
    document_content_description = "Apuntes y diapositivas de la asignatura de Estructuras de Datos."
    
    prompt = get_query_constructor_prompt(document_content_description, metadata_field_info)
    output_parser = StructuredQueryOutputParser.from_components()
    query_constructor = prompt | _llm | output_parser
    
    retriever = SelfQueryRetriever(
        query_constructor=query_constructor,
        vectorstore=vectorstore,
        verbose=True
    )
    
    st.sidebar.success("✅ Índice RAG y Retriever cargados.", icon="🧠")
    return retriever

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

# --- INICIO DE LA APP ---
st.set_page_config(page_title="Tutor ED App", layout="wide")
st.header("🤖 Tutor de Estructuras de Datos")

# --- LOGIN ---
try:
    df_estudiantes = cargar_datos_estudiantes()
    params = st.query_params
    idcv_value = params.get("idcv", [None])
    nombre_value = params.get("nombre", [None])

    if idcv_value and nombre_value:
        user_data = df_estudiantes[df_estudiantes['IDCV'] == idcv_value]
        if not user_data.empty:
            st.session_state.authenticated = True
            st.session_state.user_idcv = idcv_value
            st.session_state.user_name = user_data.iloc[0]['Nombre']
        else:
            st.error(f"❌ Usuario no autorizado. IDCV: {idcv_value}")
            st.stop()
    else:
        st.error("❌ Acceso no autorizado. Faltan credenciales en la URL.")
        st.stop()
except FileNotFoundError:
    st.error("Error crítico: El fichero de usuarios 'data/users.csv.encrypted' no se encontró en el repositorio.")
    st.stop()
except Exception as e:
    st.error(f"Ocurrió un error inesperado durante el login: {e}")
    st.stop()

# --- INICIALIZACIÓN DE SERVICIOS ---
db = get_firestore_client()
api_key = st.secrets["GOOGLE_API_KEY"]
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key, temperature=0.5)
retriever = inicializar_vectorstore_and_retriever(llm, api_key)

if retriever is None:
    st.stop()

# --- INICIO DEL CHAT ---
st.title(f"Hola, {st.session_state.user_name}")

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
            
            docs = retriever.invoke(current_prompt)
            context = "\n\n".join([d.page_content for d in docs])
            
            last5 = st.session_state.messages[-6:-1] if len(st.session_state.messages) > 1 else []
            chat_hist = "\n".join([f"- {('Pregunta' if h['role'] == 'user' else 'Respuesta')}: {h['content']}" for h in last5]) or "El estudiante no tiene interacciones previas."

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
        error_msg_to_save = {"role": "assistant", "content": error_message}
        st.session_state.messages.append(error_msg_to_save)
        save_message_to_history(db, st.session_state.user_idcv, error_msg_to_save)
    
    finally:
        st.session_state.esperando_respuesta = False
        st.rerun()