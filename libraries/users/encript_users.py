import os
import toml
from cryptography.fernet import Fernet

# === CONFIG ===
USERS_CSV_PATH = "data/users.csv"            # Archivo original (sin cifrar)
USERS_CSV_PATH_ENC = "data/users.csv.encrypted"  # Archivo encriptado
SECRETS_PATH = ".streamlit/secrets.toml"     # Ruta del secrets.toml

def cargar_clave():
    """
    Lee la clave Fernet desde .streamlit/secrets.toml
    """
    if not os.path.exists(SECRETS_PATH):
        raise FileNotFoundError(f"No se encontró {SECRETS_PATH}")
    
    secrets = toml.load(SECRETS_PATH)
    key = secrets.get("encryption", {}).get("key")
    if not key:
        raise ValueError("No se encontró la clave en [encryption] key")
    return key.encode()

def encriptar_archivo():
    """
    Encripta el archivo CSV original y genera el archivo .encrypted
    usando la clave Fernet almacenada en secrets.toml
    """
    if not os.path.exists(USERS_CSV_PATH):
        raise FileNotFoundError(f"No se encontró el archivo {USERS_CSV_PATH}")

    # Cargar clave
    fernet_key = cargar_clave()
    fernet = Fernet(fernet_key)

    # Leer contenido del CSV original
    with open(USERS_CSV_PATH, "rb") as f:
        original_data = f.read()

    # Encriptar
    encrypted_data = fernet.encrypt(original_data)

    # Guardar en archivo encriptado
    with open(USERS_CSV_PATH_ENC, "wb") as f_enc:
        f_enc.write(encrypted_data)

    print(f"✅ Archivo encriptado y guardado en {USERS_CSV_PATH_ENC}")

if __name__ == "__main__":
    encriptar_archivo()
