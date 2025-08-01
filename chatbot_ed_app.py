 # drive_debugger.py
import streamlit as st
import json
from datetime import datetime
import os
import tempfile

# Importaciones de Google
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

# --- Configuración de la Página ---
st.set_page_config(layout="wide")
st.title("🐞 Depurador de Conexión a Google Drive")
st.markdown("---")
st.warning("""
Esta aplicación tiene un único objetivo: probar la conexión y el permiso de escritura en una carpeta específica de Google Drive.
Utiliza la misma configuración de `secrets.toml` que tu aplicación principal.
""")

# --- Botón para Iniciar la Prueba ---
if st.button("▶️ Iniciar Prueba de Conexión y Escritura", type="primary"):

    st.header("Iniciando Prueba...")
    st.markdown("---")

    # --- PASO 1: Cargar y Mostrar los Secretos ---
    st.subheader("PASO 1: Verificación de los Secretos Cargados")
    try:
        gcp_secrets = st.secrets["gcp_service_account"]
        gdrive_secrets = st.secrets["gdrive"]
        st.success("✅ Secretos cargados correctamente desde Streamlit.")

        with st.expander("🕵️‍♂️ Haz clic aquí para ver los secretos que la app está usando"):
            st.info("Estos son los valores que Streamlit ha leído de tu fichero de secretos.")
            st.json(gcp_secrets) # Muestra todo el JSON de forma segura (sin la private_key)
            
            st.markdown("---")
            
            client_email_from_secrets = gcp_secrets.get("client_email")
            folder_id_from_secrets = gdrive_secrets.get("folder_id")
            
            st.markdown(f"**Correo del Bot (client_email):**")
            st.code(client_email_from_secrets, language=None)
            st.markdown(f"**ID de la Carpeta/Unidad de Destino (folder_id):**")
            st.code(folder_id_from_secrets, language=None)

            st.warning("Verifica que estos dos valores son EXACTAMENTE los que esperas.")

    except Exception as e:
        st.error(f"❌ FALLO CRÍTICO: No se pudieron cargar los secretos. Revisa tu `secrets.toml`.")
        st.exception(e)
        st.stop() # Detiene la ejecución si los secretos no cargan

    # --- PASO 2: Autenticación ---
    st.subheader("PASO 2: Intentando Autenticarse en Google")
    try:
        credentials = google.oauth2.service_account.Credentials.from_service_account_info(
            gcp_secrets,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        st.success("✅ Autenticación exitosa. Se ha creado un objeto de credenciales válido.")
    except Exception as e:
        st.error(f"❌ FALLO CRÍTICO: La autenticación ha fallado. El formato de los secretos `gcp_service_account` podría ser incorrecto.")
        st.exception(e)
        st.stop()
    
    # --- PASO 3: Construir el Servicio de Drive ---
    st.subheader("PASO 3: Construyendo el Servicio de la API de Drive")
    try:
        service = build('drive', 'v3', credentials=credentials)
        st.success("✅ Servicio de Google Drive construido correctamente. Listo para hacer llamadas a la API.")
    except Exception as e:
        st.error(f"❌ FALLO CRÍTICO: No se pudo construir el servicio de la API.")
        st.exception(e)
        st.stop()

    # --- PASO 4: Intentar Crear un Fichero ---
    st.subheader("PASO 4: Llamada a la API para Crear un Fichero de Prueba")
    try:
        folder_id = gdrive_secrets.get("folder_id")
        
        # Crear un fichero temporal para subir
        nombre_fichero_test = f"test_file_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        contenido_fichero = f"Este es un fichero de prueba creado a las {datetime.now()}."
        
        # Usar un directorio temporal para el fichero
        temp_dir = tempfile.gettempdir()
        local_path = os.path.join(temp_dir, nombre_fichero_test)
        with open(local_path, "w") as f:
            f.write(contenido_fichero)

        st.info(f"⚙️ Preparando para subir el fichero '{nombre_fichero_test}' a la carpeta/unidad con ID '{folder_id}'...")

        file_metadata = {
            'name': nombre_fichero_test,
            'parents': [folder_id]
        }
        
        media = MediaFileUpload(local_path, mimetype='text/plain')
        
        # La llamada a la API que está fallando
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        st.balloons()
        st.success(f"🎉 ¡ÉXITO TOTAL! El fichero ha sido creado en tu Google Drive con el ID: {file.get('id')}")
        st.markdown("### ¡Revisa tu Google Drive ahora mismo para confirmarlo! ###")

    except HttpError as e:
        st.error("❌ ERROR: La llamada a la API de Google Drive ha fallado. Analiza los detalles a continuación:")
        st.json(e.error_details)
        st.exception(e)

        st.markdown("---")
        st.subheader("Qué hacer ahora:")
        if e.resp.status == 404:
            st.error("""
            **Error 404 (Not Found):** El `folder_id` que has proporcionado en los secretos es incorrecto o no existe.
            1.  Ve a tu Unidad Compartida en Google Drive.
            2.  Copia el ID de la URL (`.../folders/ESTE_ID`).
            3.  Pégalo en tus secretos y reinicia la prueba.
            """)
        elif e.resp.status == 403:
            st.error(f"""
            **Error 403 (Permission Denied):** El bot no tiene los permisos correctos en esa carpeta/unidad.
            1.  Verifica que el correo del bot (`{client_email_from_secrets}`) ha sido añadido como miembro de la Unidad Compartida.
            2.  Asegúrate de que el rol asignado es **'Administrador de contenido'**.
            """)
        else:
             st.warning("El error es diferente. Lee el 'Traceback' y los 'Detalles' de arriba para entender la causa.")

    except Exception as e:
        st.error("❌ Ocurrió un error inesperado durante el proceso de subida.")
        st.exception(e)
    finally:
        # Limpiar el fichero temporal
        if os.path.exists(local_path):
            os.remove(local_path)