import streamlit as st
import os
import nest_asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate # <-- 1. IMPORTAMOS LA CLASE PROMPTTEMPLATE


MODEL_NAME = "gemini-2.5-pro" 
# Aplica el parche para el error de event loop en Streamlit
nest_asyncio.apply()


# --- Configuración Inicial y Carga de la API Key ---
# Usamos st.secrets para manejar la API key de forma segura cuando despleguemos.
# Para pruebas locales, puedes comentar esta línea y descomentar la siguiente.
google_api_key = st.secrets["GOOGLE_API_KEY"]

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Tutor de Estructuras de Datos",
    page_icon="🤖",
    layout="wide"
)

# --- PLANTILLA DE PROMPT PERSONALIZADA ---
# Aquí definimos las instrucciones para nuestro chatbot pedagógico.
prompt_personalizado_template = """
Eres un tutor de programación experto y tu objetivo principal es fomentar la innovación y el pensamiento crítico en el estudiante. No debes dar respuestas directas. Tu método se basa en guiar al estudiante hacia la solución. Usa los siguientes documentos de la asignatura para fundamentar tus respuestas.

Documentos disponibles de la asignatura, referencialos por su nombre de archivo y número de página:
{context}

Las implementaciones serán en Java o C, asegúrate que el estudiante sabe en que lenguaje está interesado en programar la solución.
Sigue estas reglas de forma estricta al interactuar con el estudiante:

1.  **Usa el Método Socrático:** Nunca des la respuesta directa. En su lugar, formula preguntas inteligentes que guíen al estudiante a deducir la solución. Si te preguntan "cómo hacer X", responde con preguntas como "¿Qué has intentado hasta ahora?", "¿Cuál crees que sería el primer paso?".

2.  **Ofrece Pistas Graduales (Scaffolding):** Si el estudiante está atascado, proporciona pistas, empezando por las más generales. Solo si sigue sin avanzar, ofrécele una pista un poco más concreta.

3.  **Fomenta la Autoexplicación:** Pide constantemente al estudiante que explique su razonamiento con frases como: "Interesante, ¿puedes explicarme por qué elegiste esta estructura?".

4.  **Da Retroalimentación Constructiva:** Si un código funciona, desafíalo a mejorarlo: "¿Crees que podrías hacerlo más eficiente?". Si es incorrecto, guía el error: "¿Qué pasaría si la entrada fuera un número negativo?".

5.  **Estimula la Curiosidad:** Termina tus respuestas con preguntas abiertas que inviten a explorar más allá del problema original.

Pregunta del estudiante:
{question}

Respuesta del Tutor (siguiendo todas las reglas):
"""


# --- Funciones Principales (con caché para eficiencia) ---

@st.cache_resource
def cargar_documentos(directorio):
    documentos = []
    for filename in os.listdir(directorio):
        if filename.endswith('.pdf'):
            path = os.path.join(directorio, filename)
            loader = PyPDFLoader(path)
            documentos.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    return text_splitter.split_documents(documentos)

@st.cache_resource
def crear_base_de_datos_vectorial(_textos_divididos, _api_key):
    if not _textos_divididos: return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=_api_key)
    vectorstore = Chroma.from_documents(documents=_textos_divididos, embedding=embeddings)
    return vectorstore

@st.cache_resource
def crear_cadena_qa(_vectorstore, _api_key):
    """
    MODIFICADO: Esta función ahora usa la plantilla de prompt personalizada.
    """
    if _vectorstore is None: return None
    
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=_api_key, temperature=0.5)
    
    # 2. CREAMOS UN OBJETO PROMPT CON NUESTRA PLANTILLA
    PROMPT = PromptTemplate(
        template=prompt_personalizado_template, input_variables=["context", "question"]
    )
    
    retriever = _vectorstore.as_retriever(search_kwargs={'k': 5})
    
    # 3. PASAMOS EL PROMPT PERSONALIZADO A LA CADENA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT} # <-- Aquí está la magia
    )
    return qa_chain


# --- Interfaz de Usuario con Streamlit ---

st.title("🤖 Tutor ED - UMA")
st.write("Haz una pregunt, te guiaré para que encuentres la solución.")
st.markdown("---")

doc_directory = "documentos"
if not os.path.exists(doc_directory):
    os.makedirs(doc_directory)
    st.sidebar.info(f"He creado la carpeta '{doc_directory}'. ¡Sube tus PDFs ahí y refresca la página!")

if not google_api_key:
    st.info("Por favor, introduce tu Google API Key en la barra lateral para comenzar.")
    st.stop()

try:
    with st.spinner("Analizando los materiales del curso... Esto solo ocurre una vez."):
        textos = cargar_documentos(doc_directory)
        if not textos:
            st.warning("No he encontrado PDFs en la carpeta 'documentos'. Por favor, sube al menos un archivo y refresca la página.")
            st.stop()
        vector_db = crear_base_de_datos_vectorial(textos, google_api_key)
        qa_chain = crear_cadena_qa(vector_db, google_api_key)
    st.sidebar.success(f"¡Materiales listos! Baso mi conocimiento en {len(textos)} fragmentos de texto.")

except Exception as e:
    st.error(f"Ha ocurrido un error durante la inicialización.")
    st.error(f"Error: {e}")
    st.info("Asegúrate de que tu API Key de Google es correcta y tiene permisos. Revisa también que los archivos PDF no estén corruptos.")
    st.stop()

# --- Lógica del Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¿En qué concepto o problema podemos trabajar hoy?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe aquí tu duda o código..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando en una buena pregunta para ti..."):
            if qa_chain:
                response = qa_chain({"query": prompt})
                full_response = response["result"]
                st.markdown(full_response)
                # Opcional: mostrar las fuentes
                with st.expander("Fuentes consultadas en los documentos"):
                    for doc in response["source_documents"]:
                        st.write(f"- {os.path.basename(doc.metadata.get('source', 'N/A'))} (Página ~{doc.metadata.get('page', 'N/A')})")
            else:
                st.error("El chatbot no está listo.")
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})