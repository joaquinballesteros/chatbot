import os
import shutil
import tempfile
from xml.dom.minidom import Document
import streamlit as st
from cryptography.fernet import Fernet
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

# --- CONSTANTES DE CIFRADO ---
PERSIST_DIR = "./.RAG/"
ENCRYPTED_FILE = "./.RAG.encrypted"

# --- FUNCIONES DE CIFRADO / DESCIFRADO ---
# Lee la clave base64 guardada en .streamlit/secrets.toml bajo [encryption]
fernet_key = st.secrets["encryption"]["key"].encode()
fernet = Fernet(fernet_key)

def encrypt_folder(src_folder: str, dest_file: str):
    # Empaqueta la carpeta src_folder en un ZIP temporal
    tmp_zip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    shutil.make_archive(tmp_zip.name.replace(".zip",""), "zip", src_folder)
    with open(tmp_zip.name, "rb") as f:
        data = f.read()
    token = fernet.encrypt(data)
    with open(dest_file, "wb") as f:
        f.write(token)
    os.unlink(tmp_zip.name)
    # Opcional: borrar la carpeta en claro
    shutil.rmtree(src_folder)

def decrypt_folder(src_file: str, dest_folder: str):
    # Si ya hay restos en claro, borrarlos
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    os.makedirs(dest_folder, exist_ok=True)
    # Lee y descrifra
    with open(src_file, "rb") as f:
        token = f.read()
    data = fernet.decrypt(token)
    tmp_zip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp_zip.write(data); tmp_zip.flush()
    shutil.unpack_archive(tmp_zip.name, dest_folder)
    os.unlink(tmp_zip.name)

@st.cache_resource
def cargar_vectorstore(api_key: str):
    """
    Carga el vectorstore desde disco usando tu embedding_function.
    """
    # 2) Configura embeddings y Chroma normalmente
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    vect = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    return vect

@st.cache_resource
def almacena_encriptado_vectorstore(api_key: str):
    """
    Almacena encriptados en la carpeta  en disco con los documentos que hay en la carpeta "Documentos".
    """
    # 1) Configura el cliente de Google Generative AI
    client = ChatGoogleGenerativeAI(
        model="models/chat-bison-001",
        google_api_key=api_key
    )

    # 2) Carga los documentos de la carpeta "Documentos"
    docs = []
    for filename in os.listdir("documentos"):
        print(f"Procesando archivo: {filename}")
        ext = os.path.splitext(filename)[1].lower()
        filepath = os.path.join("documentos", filename)
        if ext == ".pdf":
            loader = PyPDFLoader(filepath)
            docs.extend(loader.load())
        elif ext == ".md":
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                docs.append(Document(page_content=content, metadata={"source": filename}))
        else:
            # Para cualquier otro tipo de archivo de texto (txt, py, java, c, etc.)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    docs.append(Document(page_content=content, metadata={"source": filename}))
            except Exception:
                pass  # Ignora archivos que no se pueden leer como texto

    # 3) Crea el vectorstore y lo persiste
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    vect = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIR)

    vect.persist()

    # Encripta la carpeta
    encrypt_folder(PERSIST_DIR, ENCRYPTED_FILE)

    # Elimina la carpeta persistente después de encriptar
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)


    

def main():
    st.title("Generar y Encriptar Vectorstore")

    # --- CARGA DE API KEY ---
    try:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        # Elimina cualquier carpeta persistente previa
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)

        almacena_encriptado_vectorstore(google_api_key)
        st.success("Vectorstore creado y encriptado correctamente.")
        
    except (KeyError, FileNotFoundError):
        st.error("Error Crítico: La GOOGLE_API_KEY no está configurada en los secretos de Streamlit.")
        st.info("Por favor, asegúrate de añadir la clave en 'Settings > Secrets' en tu app de Streamlit Community Cloud.")
if __name__ == "__main__":
    main()

