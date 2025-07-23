import os
import shutil
import tempfile
import streamlit as st
from cryptography.fernet import Fernet
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

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

# --- INTEGRACIÓN EN STREAMLIT ---
# 1) Al inicio de la app, antes de cargar vectorstore
if os.path.exists(ENCRYPTED_FILE):
    decrypt_folder(ENCRYPTED_FILE, PERSIST_DIR)

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

# … tu lógica de autenticación y UI de Streamlit aquí …

# 3) Cada vez que inicialices o actualices y persistas:
#    - En tu script de creación (createRAG), tras vectorstore.persist():
#       encrypt_folder(PERSIST_DIR, ENCRYPTED_FILE)
#
#    - Si prefieres hacerlo al final de la sesión Streamlit, puedes:
#st.session_state.vector_store.persist()
#encrypt_folder(PERSIST_DIR, ENCRYPTED_FILE)

# --- EJEMPLO DE USO EN createRAG.py ---
def inicializar_tutor_ia(api_key):
    # (… cargado de PDFs, split, etc…)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectorstore = Chroma.from_documents(
        documents=textos_divididos,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vectorstore.persist()
    # Tras persistir, ciframos todo .RAG/ y limpiamos en claro
    encrypt_folder(PERSIST_DIR, ENCRYPTED_FILE)
    print("Vectorstore generado, cifrado y listo.")
    return vectorstore

# Ahora, tu código Streamlit puede llamar a cargar_vectorstore(st.secrets['GOOGLE_API_KEY'])
# y nunca expondrá los datos en claro en el navegador del usuario.
