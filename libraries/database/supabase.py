import os
from datetime import datetime, timezone, date
from supabase import create_client, Client
import streamlit as st

LIMIT_PER_DAY = 10
@st.cache_resource
def supa() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]  # ⚠️ solo en backend
    return create_client(url, key)


def _today():
    return date.today().isoformat()

def get_usage(db, user_id: str) -> int:
    # Garantiza la fila del día sin sumar: n = 0
    r = db.rpc("inc_usage", {"uid": user_id, "d": _today(), "n": 0}).execute()
    return int(r.data)

def increment_usage_and_stamp(db, user_id: str) -> int:
    r = db.rpc("inc_usage", {"uid": user_id, "d": _today(), "n": 1}).execute()
    return int(r.data)


def is_over_daily_limit(db: Client, user_id: str, limit=LIMIT_PER_DAY):
    used = get_usage(db, user_id)
    return (used >= limit, used)


def insert_user_if_not_exists(db: Client, user_id: str, nombre: str):
    r = db.table("users").select("*").eq("id", user_id).execute()
    if not r.data:
        db.table("users").insert({"id": user_id, "nombre": nombre}).execute()


def save_message_to_history(db, user_id, message, model: str | None = None):
    """
    Guarda un mensaje en la tabla chat_history.
    message: dict con claves 'role' y 'content'.
    model: nombre del modelo usado (guardar en mensajes del assistant).
    
    CORRECCIÓN: La columna 'content' (jsonb) debe contener el objeto completo 
    del mensaje para que load_active_user_history pueda reconstruirlo.
    """
    role = message.get("role")
    
    # ✅ CORRECCIÓN: Guardar el objeto completo en la columna content (jsonb)
    data = {
        "user_id": str(user_id),
        "role": role,  # columna role por separado para facilitar consultas
        "content": message,  # objeto completo: {"role": "...", "content": "..."}
    }
    
    if model is not None:
        data["model"] = model
    else:
        data["model"] = "unknown"

    res = db.table("chat_history").insert(data).execute()
    return res


def load_active_user_history(db: Client, user_id: str, limit: int = 10):
    # 1. Buscar el último reset del usuario
    last_reset = (
        db.table("chat_history")
          .select("created_at")
          .eq("user_id", user_id)
          .eq("role", "system")  # ✅ CORRECCIÓN: usar columna role directamente
          .order("created_at", desc=True)
          .limit(1)
          .execute()
    )

    reset_time = None
    if last_reset.data:
        reset_time = last_reset.data[0]["created_at"]

    # 2. Traer solo mensajes posteriores al último reset
    query = (
        db.table("chat_history")
          .select("content, created_at")
          .eq("user_id", user_id)
          .order("created_at", desc=True)  # primero los más nuevos
          .limit(limit)
    )

    if reset_time:
        query = query.gt("created_at", reset_time)

    r = query.execute()
    if not r.data:
        return []

    # 3. Normalizar y devolver en orden cronológico
    mensajes = []
    for row in reversed(r.data):  # invertimos porque vienen descendentes
        c = row["content"]
        if isinstance(c, dict) and "role" in c and "content" in c:
            mensajes.append(c)
        else:
            # Fallback para mensajes antiguos mal formateados
            mensajes.append({"role": "system", "content": str(c)})
    return mensajes



def reset_user_chat(db: Client, user_id: str):
    save_message_to_history(
        db,
        user_id,
        {"role": "system", "content": "--- CHAT RESETEADO POR EL USUARIO ---"}
    )
    # Si quieres limpiar todo el historial antes de marcar el reset:
    # db.table("chat_history").delete().eq("user_id", user_id).execute()

def seconds_since_last_request(db: Client, user_id: str):
    r = db.table("chat_history") \
        .select("created_at") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .limit(1) \
        .execute()
    if not r.data:
        return None
    last_ts = r.data[0]["created_at"]
    # last_ts viene como string ISO, hay que parsearlo
    dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - dt).total_seconds()