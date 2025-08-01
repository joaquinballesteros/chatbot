# firebase_debugger.py
import streamlit as st
from datetime import datetime
import google.oauth2.service_account
from google.cloud import firestore

st.set_page_config(layout="wide")
st.title("🐞 Depurador de Conexión a Firebase Firestore")
st.markdown("---")

# --- Botón para Iniciar la Prueba ---
if st.button("▶️ Iniciar Prueba de Conexión y Escritura a Firestore", type="primary"):

    st.header("Iniciando Prueba...")
    st.markdown("---")

    # --- PASO 1: Cargar y Mostrar los Secretos ---
    st.subheader("PASO 1: Verificación de los Secretos Cargados")
    try:
        key_dict = st.secrets["firebase_service_account"]
        st.success("✅ Secretos `firebase_service_account` cargados correctamente.")

        with st.expander("🕵️‍♂️ Haz clic aquí para ver los secretos que la app está usando"):
            st.info("Estos son los valores que Streamlit ha leído de tus secretos.")
            st.json({k: v for k, v in key_dict.items() if k != 'private_key'}) # Muestra todo menos la clave
            st.markdown(f"**Correo del Bot (client_email):** `{key_dict.get('client_email')}`")

    except Exception as e:
        st.error("❌ FALLO CRÍTICO: No se pudieron cargar los secretos. Revisa que `[firebase_service_account]` exista en `secrets.toml`.")
        st.exception(e)
        st.stop()

    # --- PASO 2: Autenticación y Conexión ---
    st.subheader("PASO 2: Intentando Conectarse a Firestore")
    try:
        credentials = google.oauth2.service_account.Credentials.from_service_account_info(key_dict)
        db = firestore.Client(credentials=credentials)
        st.success("✅ Conexión a Firestore exitosa. Cliente de base de datos creado.")
    except Exception as e:
        st.error("❌ FALLO CRÍTICO: La conexión a Firestore ha fallado. Revisa el formato de tus credenciales.")
        st.exception(e)
        st.stop()

    # --- PASO 3: Intentar Escribir Datos ---
    st.subheader("PASO 3: Llamada a la API para Crear un Documento de Prueba")
    try:
        # Definimos una referencia a un documento de prueba
        test_user_id = "user_debug_123"
        test_collection_ref = db.collection("test_logs")
        test_doc_ref = test_collection_ref.document(f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        st.info(f"⚙️ Preparando para escribir en la colección `test_logs`...")

        # Los datos que vamos a escribir
        test_data = {
            "message": "Prueba de escritura exitosa desde Streamlit",
            "timestamp": firestore.SERVER_TIMESTAMP, # Usa la hora del servidor de Firestore
            "user_id": test_user_id,
            "status": "SUCCESS"
        }
        
        # La llamada a la API para escribir
        test_doc_ref.set(test_data)
        
        st.balloons()
        st.success(f"🎉 ¡ÉXITO TOTAL! Los datos han sido escritos en Firestore.")
        st.markdown(f"### Revisa la colección `test_logs` en tu consola de Firebase para confirmarlo. ###")

    except Exception as e:
        st.error("❌ ERROR: La llamada a la API de Firestore para escribir ha fallado.")
        st.exception(e)
        st.markdown("---")
        st.subheader("Qué hacer ahora:")
        st.warning("""
        - **Verifica las Reglas de Seguridad:** En tu consola de Firestore, ve a la pestaña "Reglas". Para depurar, puedes ponerlas temporalmente en modo abierto (¡NO LO DEJES ASÍ EN PRODUCCIÓN!):
          ```
          rules_version = '2';
          service cloud.firestore {
            match /databases/{database}/documents {
              match /{document=**} {
                allow read, write: if true;
              }
            }
          }
          ```
        - **Verifica que la API de Firestore esté habilitada** en tu proyecto de Google Cloud.
        """)