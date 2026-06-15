#!/usr/bin/env python3

# Cuenta usuarios únicos por día en chat_history
import os
import sys
import csv
import argparse
from datetime import datetime, timedelta, timezone
from collections import defaultdict

from supabase import create_client, Client
from dotenv import load_dotenv  # 👈 importa python-dotenv

DEFAULT_PAGE_SIZE = 1000

# Carga variables del archivo .env (si existe)
load_dotenv()

def get_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("❌ No encuentro SUPABASE_URL o SUPABASE_SERVICE_ROLE_KEY en el entorno")
        sys.exit(1)
    return create_client(url, key)



def iso_utc(dt: datetime) -> str:
    # Devuelve ISO8601 en UTC con 'Z'
    return dt.astimezone(timezone.utc).replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def fetch_chat_history_rows(
    db: Client,
    since_iso: str | None,
    page_size: int = DEFAULT_PAGE_SIZE
):
    """
    Descarga todas las filas (user_id, created_at) de chat_history,
    opcionalmente filtrando por created_at >= since_iso (ISO8601).
    Pagina hasta traer todo.
    """
    table = db.table("chat_history")
    base = table.select("user_id,created_at", count="exact").order("created_at")

    if since_iso:
        base = base.gte("created_at", since_iso)

    start = 0
    rows = []
    total = None

    while True:
        resp = base.range(start, start + page_size - 1).execute()
        batch = resp.data or []
        rows.extend(batch)

        if total is None:
            # El SDK devuelve el total cuando se usa count="exact"
            total = resp.count if hasattr(resp, "count") else None

        if not batch:
            break
        start += len(batch)

        if total is not None and start >= total:
            break

    return rows


def group_daily_unique_users(rows):
    """
    rows: lista de dicts con keys 'user_id' y 'created_at'
    Devuelve dict: { 'YYYY-MM-DD': num_usuarios_unicos }
    """
    per_day_users = defaultdict(set)
    for r in rows:
        uid = str(r.get("user_id"))
        ts = r.get("created_at")
        if not uid or not ts:
            continue
        # Normaliza a fecha (UTC)
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            # Intento de parseo flexible por si llega en otro formato
            try:
                dt = datetime.strptime(str(ts).split(".")[0], "%Y-%m-%dT%H:%M:%S")
                dt = dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue
        day_str = dt.astimezone(timezone.utc).date().isoformat()
        per_day_users[day_str].add(uid)

    # Convierte sets a contadores
    return {day: len(users) for day, users in per_day_users.items()}


def print_table(daily_counts):
    if not daily_counts:
        print("No hay datos.")
        return
    print(f"{'Fecha':<12} | {'Usuarios únicos':>15}")
    print("-" * 31)
    for day in sorted(daily_counts.keys()):
        print(f"{day:<12} | {daily_counts[day]:>15}")


def write_csv(daily_counts, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "unique_users"])
        for day in sorted(daily_counts.keys()):
            w.writerow([day, daily_counts[day]])


def main():
    parser = argparse.ArgumentParser(
        description="Cuenta usuarios únicos por día en chat_history."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Limitar a los últimos N días (UTC). Si no se indica, consulta todo."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Ruta de salida CSV (opcional)."
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help=f"Tamaño de página para la descarga (por defecto {DEFAULT_PAGE_SIZE})."
    )
    args = parser.parse_args()

    db = get_client()
    since_iso = None
    if args.days and args.days > 0:
        since_dt = datetime.now(timezone.utc) - timedelta(days=args.days)
        since_iso = iso_utc(since_dt)

    rows = fetch_chat_history_rows(db, since_iso, page_size=args.page_size)
    daily_counts = group_daily_unique_users(rows)

    print_table(daily_counts)

    if args.csv:
        write_csv(daily_counts, args.csv)
        print(f"\nCSV guardado en: {args.csv}")


if __name__ == "__main__":
    main()
