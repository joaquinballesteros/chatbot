import os
import shutil
import tempfile
from cryptography.fernet import Fernet
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# --- CONSTANTES DE CIFRADO ---
import streamlit as st
fernet_key = st.secrets["encryption"]["key"].encode()
fernet = Fernet(fernet_key)

# Funciones de cifrado/descifrado de archivos y carpetas

def encrypt_file(src_path: str, dest_path: str):
    """
    Encripta un archivo individual y elimina el original.
    """
    data = open(src_path, 'rb').read()
    token = fernet.encrypt(data)
    with open(dest_path, 'wb') as f:
        f.write(token)
    os.remove(src_path)


def encrypt_folder(src_folder: str, dest_file: str):
    """
    Encripta una carpeta en un único fichero .encrypted.
    """
    tmp_zip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    shutil.make_archive(tmp_zip.name.replace('.zip',''), 'zip', src_folder)
    data = open(tmp_zip.name, 'rb').read()
    token = fernet.encrypt(data)
    with open(dest_file, 'wb') as f:
        f.write(token)
    os.unlink(tmp_zip.name)
    shutil.rmtree(src_folder)


def decrypt_folder(src_file: str, dest_folder: str):
    """
    Desencripta el fichero .encrypted a una carpeta.
    """
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    os.makedirs(dest_folder, exist_ok=True)
    token = open(src_file, 'rb').read()
    data = fernet.decrypt(token)
    tmp_zip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp_zip.write(data)
    tmp_zip.flush()
    shutil.unpack_archive(tmp_zip.name, dest_folder)
    os.unlink(tmp_zip.name)

# Generación y persistencia del vectorstore
PERSIST_DIR = "./.RAG/"
ENCRYPTED_STORE = "./data/.RAG.encrypted"
CSV_USERS_PATH = os.path.join("data", "users.csv")
CSV_USERS_ENC = os.path.join("data", "users.csv.encrypted")


def almacena_encriptado_vectorstore(api_key: str):
    """
    Crea el vectorstore a partir de documentos y persiste cifrado.
    """
    # 1) Carga documentos
    docs = []
    for fn in os.listdir("documentos"):
        path = os.path.join("documentos", fn)
        if fn.lower().endswith('.pdf'):
            docs.extend(PyPDFLoader(path).load())
        else:
            try:
                content = open(path, 'r', encoding='utf-8').read()
                docs.append(Document(page_content=content, metadata={"source": fn}))
            except Exception:
                continue

    # 2) Construye y persiste vectorstore en claro
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vect = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIR)
    vect.persist()

    # 3) Cifra la carpeta del vectorstore
    encrypt_folder(PERSIST_DIR, ENCRYPTED_STORE)


def main():
    """
    Punto de entrada para generar el vectorstore y cifrar también el CSV de usuarios.
    """
    import streamlit as st
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        raise RuntimeError("No se encontró GOOGLE_API_KEY en secretos.")

    # 1) Genera y encripta el vectorstore
    almacena_encriptado_vectorstore(api_key)

    # 2) Encripta el CSV de usuarios si existe
    if os.path.exists(CSV_USERS_PATH):
        encrypt_file(CSV_USERS_PATH, CSV_USERS_ENC)
        print(f"CSV de usuarios cifrado en {CSV_USERS_ENC}")
    else:
        print(f"No se encontró {CSV_USERS_PATH}, se omite cifrado de usuarios.")


if __name__ == "__main__":
    main()
