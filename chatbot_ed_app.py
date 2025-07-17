import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# --- Configuración Inicial y Carga de la API Key ---
# Usamos st.secrets para manejar la API key de forma segura cuando despleguemos.
# Para pruebas locales, puedes comentar esta línea y descomentar la siguiente.
# google_api_key = st.secrets["GOOGLE_API_KEY"]

# Para pruebas locales, crea un archivo .env con GOOGLE_API_KEY="TU_API_KEY"
# from dotenv import load_dotenv
# load_dotenv()
# google_api_key = os.getenv("GOOGLE_API_KEY")

# Esta es una forma simple de pedir la key en la UI si no quieres usar secretos
google_api_key = st.sidebar.text_input("Introduce tu Google API Key", type="password")


# CONFIGURACIÓN DE LA PÁGINA (Añade esto al principio)
st.set_page_config(
    page_title="Chatbot de Programación",
    page_icon="🤖",
    layout="wide"
)

# --- El resto de tu código sigue aquí ---
st.title("🤖 Chatbot para tu Asignatura de Programación")


# --- Funciones Principales (con caché para eficiencia) ---

@st.cache_resource
def cargar_documentos(directorio):
    """Carga y procesa todos los PDFs de un directorio."""
    documentos = []
    for filename in os.listdir(directorio):
        if filename.endswith('.pdf'):
            path = os.path.join(directorio, filename)
            loader = PyPDFLoader(path)
            documentos.extend(loader.load())
    
    # Dividir los documentos en trozos más pequeños
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    textos_divididos = text_splitter.split_documents(documentos)
    return textos_divididos

@st.cache_resource
def crear_base_de_datos_vectorial(_textos_divididos, _api_key):
    """Crea la base de datos de vectores usando los embeddings de Google."""
    if not _textos_divididos:
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=_api_key)
    # Chroma.from_documents es efímero. Para persistencia, usa persist_directory.
    vectorstore = Chroma.from_documents(documents=_textos_divididos, embedding=embeddings)
    return vectorstore

@st.cache_resource
def crear_cadena_qa(_vectorstore, _api_key):
    """Crea la cadena de preguntas y respuestas (RAG)."""
    if _vectorstore is None:
        return None
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=_api_key, temperature=0.3)
    
    # Creamos el "retriever" que busca en la base de datos vectorial
    retriever = _vectorstore.as_retriever()
    
    # Creamos la cadena que une el retriever y el LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain


# --- Interfaz de Usuario con Streamlit ---

st.title("🤖 Chatbot para tu Asignatura de Programación")
st.write("Este chatbot responderá preguntas basándose en los documentos que has subido.")

# Directorio de documentos
doc_directory = "documentos"
if not os.path.exists(doc_directory):
    os.makedirs(doc_directory)
    st.sidebar.info(f"He creado la carpeta '{doc_directory}'. ¡Sube tus PDFs ahí!")

# Verifica si la API key ha sido introducida
if not google_api_key:
    st.info("Por favor, introduce tu Google API Key en la barra lateral para comenzar.")
    st.stop()

# Cargar documentos y crear la base de datos
try:
    with st.spinner("Procesando documentos... Esto puede tardar unos minutos la primera vez."):
        textos = cargar_documentos(doc_directory)
        if not textos:
            st.warning("No se encontraron documentos PDF en la carpeta 'documentos'. Por favor, sube al menos un archivo.")
            st.stop()
            
        vector_db = crear_base_de_datos_vectorial(textos, google_api_key)
        qa_chain = crear_cadena_qa(vector_db, google_api_key)
    st.sidebar.success(f"¡Listo! {len(textos)} trozos de texto procesados.")

except Exception as e:
    st.error(f"Ha ocurrido un error durante la inicialización: {e}")
    st.info("Asegúrate de que tu API Key de Google es correcta y tiene permisos.")
    st.stop()


# --- Lógica del Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input del usuario
if prompt := st.chat_input("¿Cuál es tu pregunta sobre la asignatura?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta del asistente
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            if qa_chain:
                response = qa_chain({"query": prompt})
                st.markdown(response["result"])
                # Opcional: mostrar las fuentes que usó para responder
                with st.expander("Fuentes utilizadas"):
                    for doc in response["source_documents"]:
                        st.write(f"- Página {doc.metadata.get('page', 'N/A')} de {os.path.basename(doc.metadata.get('source', 'N/A'))}")
            else:
                st.error("El chatbot no está listo. Revisa los pasos anteriores.")
    
    st.session_state.messages.append({"role": "assistant", "content": response["result"]})