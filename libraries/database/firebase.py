
import streamlit as st
import google.oauth2.service_account
from google.cloud import firestore
from datetime import datetime, timezone
import time
# --- LÍMITE DE USO ---
LIMIT_PER_DAY = 30          # nº máximo de preguntas por día/usuario

def _usage_ref(db, user_id):
    return db.collection("usage_limits").document(user_id)

def _today_iso_utc():
    return datetime.now(timezone.utc).date().isoformat()

def get_usage(db, user_id):
    doc = _usage_ref(db, user_id).get()
    return doc.to_dict() or {}

def is_over_daily_limit(db, user_id, limit=LIMIT_PER_DAY):
    data = get_usage(db, user_id)
    today = _today_iso_utc()
    used = int(data.get("counts", {}).get(today, 0))
    return (used >= limit, used)

def _inc_usage_tr(tr, ref, today):
    snap = tr.get(ref)
    data = snap.to_dict() or {}
    counts = data.get("counts", {})
    counts[today] = int(counts.get(today, 0)) + 1
    tr.set(ref, {
        "counts": counts,
        "last_request_at": datetime.now(timezone.utc).timestamp()
    }, merge=True)

def increment_usage_and_stamp(db, user_id):
    today = _today_iso_utc()
    ref = _usage_ref(db, user_id)
    now_epoch = datetime.now(timezone.utc).timestamp()
    try:
        # Usamos `update` para incrementar el contador de forma atómica
        ref.update({
            "last_request_at": now_epoch,
            f"counts.{today}": firestore.Increment(1),
        })
    except Exception:
        # Si no existe, lo creamos
        ref.set({
            "last_request_at": now_epoch,
            "counts": {today: 1},
        }, merge=True)


def seconds_since_last_request(db, user_id):
    data = get_usage(db, user_id)
    ts = data.get("last_request_at")
    if not ts:
        return None
    return time.time() - float(ts)



def get_firestore_client_chatbot():
    key_dict = st.secrets["firebase_service_account"]
    creds = google.oauth2.service_account.Credentials.from_service_account_info(key_dict)
    db = firestore.Client(credentials=creds)
    return db

def load_active_user_history(db: firestore.Client, user_id: str):
    historial_ref = db.collection('users').document(user_id).collection('historial')
    reset_query = historial_ref.where("role", "==", "system").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1)
    reset_docs = list(reset_query.stream())
    last_reset_timestamp = reset_docs[0].get('timestamp') if reset_docs else None
    query = (
        historial_ref.where("timestamp", ">", last_reset_timestamp).order_by("timestamp")
        if last_reset_timestamp else historial_ref.order_by("timestamp")
    )
    return [doc.to_dict() for doc in query.stream()]

def save_message_to_history(db: firestore.Client, user_id: str, message: dict):
    message_to_save = message.copy()
    message_to_save['timestamp'] = firestore.SERVER_TIMESTAMP
    db.collection('users').document(user_id).collection('historial').add(message_to_save)

def reset_user_chat(db: firestore.Client, user_id: str):
    reset_message = {"role": "system", "content": f"--- CHAT RESETEADO POR EL USUARIO ---"}
    save_message_to_history(db, user_id, reset_message)
    st.session_state.messages = []
