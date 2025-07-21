import streamlit as st
import os
import pandas as pd
import json
from datetime import datetime, timezone
import nest_asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# --- CONFIGURACIÓN DE PÁGINA ---
# Se llama una única vez y al principio de todo.
st.set_page_config(
    page_title="Tutor de Estructuras de Datos",
    page_icon="🤖",
    layout="wide"
)

# --- CONSTANTES ---
MODEL_NAME = "gemini-2.5-pro"
DOCS_DIRECTORY = "documentos"
STUDENT_DATA_DIRECTORY = "estudiantes"
HISTORY_DIRECTORY = "historiales"
HISTORY_FILE = os.path.join(HISTORY_DIRECTORY, "user_profiles.json")

nest_asyncio.apply()

# --- CARGA DE API KEY ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Error Crítico: La GOOGLE_API_KEY no está configurada en los secretos de Streamlit.")
    st.info("Por favor, asegúrate de añadir la clave en 'Settings > Secrets' en tu app de Streamlit Community Cloud.")
    st.stop()


# --- FUNCIONES DE GESTIÓN DE HISTORIAL ---
def cargar_o_crear_historiales():
    if not os.path.exists(HISTORY_DIRECTORY):
        os.makedirs(HISTORY_DIRECTORY)
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        return {}
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def guardar_historial(datos):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(datos, f, ensure_ascii=False, indent=4)

def formatear_historial_para_prompt(historial_usuario, n_ultimos=5):
    if not historial_usuario:
        return "El estudiante no tiene interacciones previas."
    interacciones_recientes = historial_usuario[-n_ultimos:]
    historial_formateado = "\n".join([
        f"- Pregunta anterior: \"{item['prompt']}\"\n  Respuesta del tutor: \"{item['respuesta']}\""
        for item in interacciones_recientes
    ])
    return f"A continuación se muestran las interacciones más recientes del estudiante:\n{historial_formateado}"

# --- FUNCIONES DE AUTENTICACIÓN ---
@st.cache_data(ttl=3600)
def cargar_datos_estudiantes(directorio):
    try:
        archivos_csv = [f for f in os.listdir(directorio) if f.endswith('.csv')]
        if not archivos_csv:
            st.error(f"Error de configuración: No se encontró ningún fichero CSV en la carpeta '{directorio}'.")
            return None
        ruta_csv = os.path.join(directorio, archivos_csv[0])
        df = pd.read_csv(ruta_csv)
        df['IDCV'] = df['IDCV'].astype(str).str.strip()
        df['DNI/NIE/Pasaporte'] = df['DNI/NIE/Pasaporte'].astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Error al cargar fichero de estudiantes: {e}")
        return None

def verificar_credenciales(df_estudiantes, codigo_acceso):
    if df_estudiantes is None or not codigo_acceso:
        return None, None
    codigo_acceso = str(codigo_acceso).strip()
    usuario = df_estudiantes[df_estudiantes['IDCV'] == codigo_acceso]
    if not usuario.empty:
        return usuario.iloc[0]['IDCV'], usuario.iloc[0]['Nombre']
    usuario = df_estudiantes[df_estudiantes['DNI/NIE/Pasaporte'] == codigo_acceso]
    if not usuario.empty:
        return usuario.iloc[0]['IDCV'], usuario.iloc[0]['Nombre']
    return None, None

# --- FUNCIONES DEL TUTOR DE IA ---
@st.cache_resource
def inicializar_tutor_ia(_api_key):
    try:
        if not os.path.exists(DOCS_DIRECTORY) or not any(f.endswith('.pdf') for f in os.listdir(DOCS_DIRECTORY)):
            st.error(f"Error: La carpeta '{DOCS_DIRECTORY}' no existe o no contiene archivos PDF.")
            return None
        documentos = []
        for filename in os.listdir(DOCS_DIRECTORY):
            if filename.endswith('.pdf'):
                path = os.path.join(DOCS_DIRECTORY, filename)
                loader = PyPDFLoader(path)
                documentos.extend(loader.load())
        if not documentos:
            return None
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
        textos_divididos = text_splitter.split_documents(documentos)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=_api_key)
        vectorstore = Chroma.from_documents(documents=textos_divididos, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error crítico al inicializar el vector store: {e}")
        return None

# --- PLANTILLA DE PROMPT ---
prompt_personalizado_template_con_historial = """
Eres un tutor de programación experto y tu objetivo es personalizar la asistencia basándote en el historial del estudiante para fomentar la innovación y el pensamiento crítico. No debes dar respuestas directas. Tu método se basa en guiar al estudiante hacia la solución.

**Contexto del Historial del Estudiante:**
Aquí tienes un resumen de las últimas preguntas del estudiante. Úsalo para entender su nivel actual y adaptar tus preguntas. Si ves que pregunta repetidamente sobre un tema, quizás necesite un enfoque diferente o una pista más fundamental.
{chat_history}

**Documentos de la Asignatura:**
Usa los siguientes documentos para fundamentar tus respuestas, referenciandolos por su nombre de archivo y página.
{context}

**Reglas Estrictas de Interacción:**
1.  **Usa el Método Socrático:** Nunca des la respuesta directa. Guía con preguntas.
2.  **Adapta la Dificultad:** Usa el historial para juzgar si tus preguntas son muy fáciles o muy difíciles.
3.  **Fomenta la Autoexplicación:** Pide que explique su razonamiento.
4.  **Retroalimentación Constructiva y Personalizada:** Si comete un error similar a uno anterior (visible en el historial), puedes decir algo como: "Veo que de nuevo estamos trabajando con punteros. ¿Recuerdas qué pasaba la última vez que intentamos acceder a memoria no inicializada?".
5.  **Estimula la Curiosidad:** Termina con preguntas abiertas.

**Implementación:**
Las implementaciones serán en Java o C. Asegúrate de que el estudiante especifica el lenguaje.

**Pregunta Actual del Estudiante:**
{question}

**Respuesta del Tutor Personalizada (siguiendo todas las reglas):**
"""

# --- LÓGICA PRINCIPAL DE LA APLICACIÓN ---

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# PANTALLA DE LOGIN
if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.header("🤖 Tutor de Estructuras de Datos")
        st.write("Por favor, identifícate con tu DNI o IDCV para acceder.")
        for dir_path in [STUDENT_DATA_DIRECTORY, DOCS_DIRECTORY, HISTORY_DIRECTORY]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        df_estudiantes = cargar_datos_estudiantes(STUDENT_DATA_DIRECTORY)
        if df_estudiantes is not None:
            with st.form("login_form"):
                codigo_acceso = st.text_input("DNI / IDCV", placeholder="Introduce tu identificador")
                submitted = st.form_submit_button("Acceder")
                if submitted:
                    idcv, nombre_usuario = verificar_credenciales(df_estudiantes, codigo_acceso)
                    if nombre_usuario:
                        st.session_state.authenticated = True
                        st.session_state.user_name = nombre_usuario
                        st.session_state.user_idcv = idcv
                        with st.spinner("Preparando el entorno del tutor..."):
                            st.session_state.vector_store = inicializar_tutor_ia(google_api_key)
                        st.rerun()
                    else:
                        st.error("Identificador no encontrado.")

# PANTALLA DEL CHATBOT
else:
    st.title(f"🤖 Tutor ED - UMA")
    st.write(f"Hola **{st.session_state.user_name}**, haz una pregunta. Te guiaré para que encuentres la solución.")
    st.markdown("---")
    
    vector_store = st.session_state.get('vector_store')
    if not vector_store:
        st.error("El tutor no pudo inicializarse correctamente. Contacta al administrador.")
        st.stop()
        
    user_profiles = cargar_o_crear_historiales()
    user_idcv = st.session_state.user_idcv
    if user_idcv not in user_profiles:
        user_profiles[user_idcv] = {"nombre": st.session_state.user_name, "historial": []}
    current_user_history = user_profiles[user_idcv]["historial"]

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "¿Sobre qué tema o estructura de datos tienes dudas hoy?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Escribe aquí tu duda o código..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Buscando documentos y personalizando tu pregunta..."):
                
                # Paso 1: Recuperar documentos relevantes
                retriever = vector_store.as_retriever(search_kwargs={'k': 5})
                retrieved_docs = retriever.get_relevant_documents(prompt)
                
                # Formatear el contexto de los documentos para el prompt
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # Paso 2: Formatear el historial del chat
                historial_formateado = formatear_historial_para_prompt(current_user_history)
                
                # Paso 3: Crear la plantilla y la cadena LLM
                prompt_template = PromptTemplate(
                    template=prompt_personalizado_template_con_historial, 
                    input_variables=["context", "question", "chat_history"]
                )
                llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=google_api_key, temperature=0.5, convert_system_message_to_human=True)
                llm_chain = LLMChain(llm=llm, prompt=prompt_template)
                
                # Paso 4: Ejecutar la cadena con TODAS las variables necesarias
                response = llm_chain.invoke({
                    "context": context_text,
                    "question": prompt,
                    "chat_history": historial_formateado
                })
                
                full_response = response.get("text", "Lo siento, he encontrado un problema al generar la respuesta.")

                # Guardar la nueva interacción en el historial
                nueva_interaccion = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompt": prompt,
                    "respuesta": full_response
                }
                user_profiles[user_idcv]["historial"].append(nueva_interaccion)
                guardar_historial(user_profiles)

                st.markdown(full_response)
                
                # Mostrar las fuentes
                with st.expander("Fuentes consultadas"):
                    if retrieved_docs:
                        for doc in retrieved_docs:
                            page_num = doc.metadata.get('page', -1) + 1
                            st.write(f"- Fichero: {os.path.basename(doc.metadata.get('source', 'N/A'))}, Página: ~{page_num}")
                    else:
                        st.write("No se encontraron documentos relevantes para esta pregunta.")
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})