
import os
import re
import pandas as pd
import streamlit as st
from cryptography.fernet import Fernet
# === CONFIG ===
USERS_CSV_PATH_ENC = "data/users.csv.encrypted"
TMP_DIR = "tmp"
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)
USERS_CSV_PATH_TMP = os.path.join(TMP_DIR, "users.csv.tmp")


@st.cache_data(ttl=600)
def cargar_datos_estudiantes():
    """
    Loads and decrypts student data from an encrypted CSV file.

    This function reads an encrypted CSV file containing student information, decrypts it using a Fernet key
    retrieved from Streamlit secrets, and loads the data into a pandas DataFrame. The decrypted file is temporarily
    written to disk and removed after loading. The 'IDCV' column values are stripped of leading and trailing whitespace.

    Returns:
        pandas.DataFrame: A DataFrame containing the student data with all columns as strings and cleaned 'IDCV' values.
    """
    fernet_key = st.secrets["encryption"]["key"].encode()
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