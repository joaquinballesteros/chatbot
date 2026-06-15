#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import argparse
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

from dotenv import load_dotenv
from supabase import create_client, Client

# Exporta todo el diálogo (preguntas, respuestas, fechas y modelo) por usuario a out_dialogs/


# ================== Config ==================
load_dotenv()

DEFAULT_PAGE_SIZE = 2000
OUTDIR_DEFAULT = "estadisticas/out_dialogs"

# ================== Helpers ==================

def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        print("❌ Configura SUPABASE_URL y SUPABASE_SERVICE_ROLE_KEY (o ANON).")
        sys.exit(1)
    return create_client(url, key)

def to_utc_start(date_str: str) -> str:
    dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")

def to_utc_exclusive_end(date_str: str) -> str:
    dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc) + timedelta(days=1)
    return dt.isoformat().replace("+00:00", "Z")

def safe_filename(s: str) -> str:
    keep = "._-"
    return "".join(ch if ch.isalnum() or ch in keep else "_" for ch in str(s))[:100]

# ================== DB ==================

def fetch_all_users(db: Client) -> List[Dict[str, Any]]:
    # Solo recuperamos id (que es tu idcv)
    r = db.table("users").select("id").order("id").execute()
    return r.data or []

def fetch_user_dialog_between(
    db: Client,
    user_id: str,
    start_iso: str,
    end_iso_exclusive: str,
    page_size: int = DEFAULT_PAGE_SIZE,
    include_system: bool = False
) -> List[Dict[str, Any]]:
    base = (
        db.table("chat_history")
          .select("user_id,created_at,content,model")
          .eq("user_id", user_id)
          .gte("created_at", start_iso)
          .lt("created_at", end_iso_exclusive)
          .order("created_at")
    )

    start = 0
    rows: List[Dict[str, Any]] = []
    while True:
        resp = base.range(start, start + page_size - 1).execute()
        batch = resp.data or []
        if not batch:
            break
        for r in batch:
            c = r.get("content")
            if isinstance(c, dict):
                role = c.get("role")
                text = c.get("content")
            else:
                role, text = "system", str(c) if c is not None else ""
            if not include_system and role == "system":
                continue
            rows.append({
                "created_at": r.get("created_at"),
                "role": role or "",
                "content": text or "",
                "model": r.get("model") or "",
            })
        if len(batch) < page_size:
            break
        start += len(batch)
    return rows

# ================== Turn grouping ==================

def assign_turn_ids(rows: List[Dict[str, Any]]) -> None:
    turn = 0
    for r in rows:
        if r["role"] == "user":
            turn += 1
        r["turn_id"] = turn

# ================== Writers ==================

def write_user_csv(
    outdir: str,
    user_idcv: str,
    rows: List[Dict[str, Any]]
) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{safe_filename(user_idcv)}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["created_at", "role", "content", "turn_id", "model"])
        for r in rows:
            w.writerow([
                r.get("created_at", ""),
                r.get("role", ""),
                r.get("content", ""),
                r.get("turn_id", 0),
                r.get("model", ""),
            ])
    return path

# ================== MAIN ==================

def main():
    parser = argparse.ArgumentParser(
        description="Exporta por usuario todo el diálogo (preguntas, respuestas, fechas y modelo) para resumir con un LLM."
    )
    parser.add_argument("--from", dest="from_date", required=True, help="Fecha inicio (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", required=True, help="Fecha fin (YYYY-MM-DD, inclusive)")
    parser.add_argument("--outdir", type=str, default=OUTDIR_DEFAULT, help="Directorio de salida")
    parser.add_argument("--include-system", action="store_true", help="Incluir mensajes 'system'")
    args = parser.parse_args()

    db = get_supabase_client()
    start_iso = to_utc_start(args.from_date)
    end_iso_exclusive = to_utc_exclusive_end(args.to_date)

    users = fetch_all_users(db)
    if not users:
        print("No hay usuarios en 'users'.")
        return

    print(f"Exportando diálogos de {len(users)} usuarios · Rango {args.from_date} → {args.to_date} (inclusive)\n")

    exported = 0
    for u in users:
        user_idcv = str(u.get("id"))  # el idcv
        rows = fetch_user_dialog_between(db, user_idcv, start_iso, end_iso_exclusive, include_system=args.include_system)

        if not rows:
            print(f"- {user_idcv}: sin mensajes en el rango.")
            continue

        assign_turn_ids(rows)
        csv_path = write_user_csv(args.outdir, user_idcv, rows)
        exported += 1
        print(f"- {user_idcv}: {len(rows)} mensajes → {csv_path}")

    if exported == 0:
        print("\nNo se exportó ningún usuario.")
    else:
        print(f"\n✅ Listo. Archivos en: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
