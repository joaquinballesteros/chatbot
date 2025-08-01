# 1_⚙️_Admin_-_Crear_Indice.py

import streamlit as st
import os
import time

# --- Librerías de LangChain y Google ---
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import asyncio

# --- Rutas ---
PDFS_PATH = "documentos"
INDEX_PATH = "faiss_index"

# --- Función para crear el ---
def create_vectorstore(api_key):
    """
    Lee PDFs, crea embeddings y guarda el índice FAISS.
    Muestra el progreso en la interfaz de Streamlit.
    """
    st.info("Iniciando la creación del vectorstore...")
    progress_bar = st.progress(0, text="Paso 1/5: Verificando documentos PDF...")
    time.sleep(1)

    # 1. Verificar y Cargar los documentos PDF
    if not os.path.exists(PDFS_PATH) or not os.listdir(PDFS_PATH):
        st.error(f"Error: La carpeta '{PDFS_PATH}' no existe o está vacía en el repositorio.")
        st.info("Por favor, sube tus ficheros PDF a esa carpeta en GitHub y vuelve a desplegar la app.")
        progress_bar.empty()
        return
        
    loader = PyPDFDirectoryLoader(PDFS_PATH)
    documents = loader.load()
    st.write(f"Se han cargado {len(documents)} documentos desde la carpeta `{PDFS_PATH}`.")
    progress_bar.progress(20, text="Paso 2/5: Dividiendo documentos en trozos...")
    time.sleep(1)

    # 2. Dividir los documentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    st.write(f"Los documentos han sido divididos en {len(docs)} trozos (chunks).")
    progress_bar.progress(40, text="Paso 3/5: Inicializando modelo de embeddings...")
    time.sleep(1)

    # 3. Inicializar el modelo de embeddings
    try:
        # Arreglo para el bucle de eventos de asyncio
        asyncio.set_event_loop(asyncio.new_event_loop())
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
    except Exception as e:
        st.error(f"Error al inicializar el modelo de embeddings: {e}")
        progress_bar.empty()
        return
    st.write("Modelo de embeddings inicializado.")
    progress_bar.progress(60, text="Paso 4/5: Creando el índice FAISS (puede tardar)...")

    # 4. Crear el índice FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)
    progress_bar.progress(80, text="Paso 5/5: Guardando el índice en el disco...")
    time.sleep(1)

    # 5. Guardar el índice
    # ¡OJO! Esto guarda el índice en el contenedor temporal de Streamlit.
    # No es persistente entre reinicios, pero sí para la sesión actual y futuras
    # si el contenedor no se recicla. El objetivo es crear el índice para que
    # la app principal lo pueda usar.
    vectorstore.save_local(INDEX_PATH)
    
    progress_bar.progress(100, text="¡Índice creado y guardado con éxito!")
    st.success(f"¡Éxito! El índice ha sido guardado en la ruta: `{INDEX_PATH}`.")
    st.info("Ahora puedes ir a la página principal del chatbot.")

# --- Interfaz de la Página ---
st.title("⚙️ Panel de Administración")
st.header("Creación del Índice Vectorial (RAG)")

st.warning("""
**¡Atención!** Este proceso lee todos los PDFs de la carpeta `documentos_pdf`
y crea un nuevo índice vectorial. Esto puede tardar varios minutos y consumirá
recursos de la API de Google. Solo ejecútalo cuando hayas actualizado los PDFs.
""", icon="⚠️")

# --- Formulario de Contraseña ---
# Usamos un secreto para la contraseña del panel de admin
admin_password = st.secrets.get("app_config", {}).get("admin_idcv", "")

if not admin_password:
    st.error("No se ha configurado una contraseña de administrador en los secretos.")
    st.stop()

password = st.text_input("Ingresa la contraseña para continuar:", type="password")

if password == admin_password:
    st.success("Acceso concedido.", icon="✅")
    
    if st.button("🚀 Crear y Guardar el Índice FAISS", type="primary"):
        with st.status("Procesando documentos...", expanded=True) as status:
            try:
                api_key = st.secrets["GOOGLE_API_KEY"]
                create_vectorstore(api_key)
                status.update(label="¡Proceso completado!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Ocurrió un error inesperado: {e}")
                st.code(traceback.format_exc())
                status.update(label="Proceso fallido", state="error")
else:
    if password:
        st.error("Contraseña incorrecta.", icon="❌")