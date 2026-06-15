#!/usr/bin/env python3
import os
import sys
import csv
import argparse
from datetime import datetime, timedelta, timezone
from supabase import create_client, Client
from dotenv import load_dotenv

# Cargar variables del .env
load_dotenv()

def get_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("❌ Faltan SUPABASE_URL o SUPABASE_SERVICE_ROLE_KEY")
        sys.exit(1)
    return create_client(url, key)


def fetch_user_questions(db: Client, from_date: str, to_date: str):
    """
    Recupera todas las preguntas (role=user) entre dos fechas UTC.
    Las fechas deben venir en formato YYYY-MM-DD.
    Devuelve una lista de dicts con: user_id, created_at, question, model
    """
    # Convertir a límites ISO8601
    from_iso = datetime.fromisoformat(from_date).replace(tzinfo=timezone.utc).isoformat()
    to_date_obj = datetime.fromisoformat(to_date).replace(tzinfo=timezone.utc)
    to_iso = (to_date_obj + timedelta(days=1)).isoformat()

    resp = (
        db.table("chat_history")
        .select("user_id, created_at, content, model")
        .gte("created_at", from_iso)
        .lte("created_at", to_iso)
        .order("created_at")
        .execute()
    )

    rows = resp.data or []
    preguntas = []

    for r in rows:
        c = r.get("content")
        if isinstance(c, dict):
            role = c.get("role")
            text = c.get("content")
        else:
            role, text = None, None

        if role == "user" and text:
            preguntas.append({
                "user_id": r.get("user_id"),
                "created_at": r.get("created_at"),
                "question": text,
                # El modelo puede ser None si solo se guarda en respuestas del assistant
                "model": r.get("model") or ""
            })

    return preguntas


def print_questions(preguntas):
    if not preguntas:
        print("No se encontraron preguntas en ese rango de fechas.")
        return

    for p in preguntas:
        model_txt = f" | model={p['model']}" if p.get("model") else ""
        print(f"[{p['created_at']}] Usuario {p['user_id']} → {p['question']}{model_txt}")


def write_csv(preguntas, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["user_id", "created_at", "question", "model"])
        w.writeheader()
        w.writerows(preguntas)
    print(f"✅ CSV guardado en {path}")


def main():
    parser = argparse.ArgumentParser(description="Extrae todas las preguntas entre dos fechas")
    parser.add_argument("--from", dest="from_date", required=True, help="Fecha inicial (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", required=True, help="Fecha final (YYYY-MM-DD)")
    parser.add_argument("--csv", type=str, default=None, help="Ruta para guardar CSV (opcional)")
    args = parser.parse_args()

    db = get_client()
    preguntas = fetch_user_questions(db, args.from_date, args.to_date)

    print_questions(preguntas)

    if args.csv:
        write_csv(preguntas, args.csv)


if __name__ == "__main__":
    main()
