#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
from collections import Counter

import pandas as pd
from dotenv import load_dotenv

# OpenAI SDK >= 1.0
try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("Falta 'openai'. Instala: pip install openai python-dotenv pandas")

# =========================
# Config
# =========================

load_dotenv()  # lee .env (OPENAI_API_KEY)

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_IN_DIR = "estadisticas/out_dialogs"
DEFAULT_OUT_DIR = "estadisticas/out_summaries"

# límites para no reventar tokens
MAX_TURNS_FOR_LLM = 250            # máx. turnos (preguntas) a enviar al LLM
MAX_ANSWER_CHARS_IN_PROMPT = 300   # recorte de la primera respuesta por turno


# =========================
# Utilidades
# =========================

def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def idcv_from_filename(path: str) -> str:
    """Extrae el idcv del nombre de fichero (p.ej., 'ABC123.csv' -> 'ABC123')."""
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else (s[:n] + "…")


# =========================
# Carga y preprocesado
# =========================

def load_user_dialog(csv_path: str) -> pd.DataFrame:
    """
    Espera columnas: created_at, role, content, turn_id
    Puede incluir opcionalmente 'model'.
    """
    df = pd.read_csv(csv_path, dtype={"role": "string", "content": "string", "model": "string"})
    for col in ["created_at", "role", "content", "turn_id"]:
        if col not in df.columns:
            raise ValueError(f"{csv_path}: falta columna '{col}'")
    # Asegura orden cronológico
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df = df.sort_values("created_at", kind="mergesort").reset_index(drop=True)
    # Limpieza mínima
    df["role"] = df["role"].fillna("").str.lower()
    df["content"] = df["content"].fillna("").map(normalize_text)
    if "model" not in df.columns:
        df["model"] = ""
    else:
        df["model"] = df["model"].fillna("").astype(str)
    return df


def extract_turns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Devuelve una lista de turnos:
    [
      {
        "turn_id": 1,
        "question": "...",
        "question_time": "...Z",
        "answer": "... (primera respuesta del asistente recortada)",
        "answer_time": "...Z" (si existe),
        "answer_model": "gpt-4o-mini" (si existe)
      },
      ...
    ]
    """
    turns: List[Dict[str, Any]] = []
    if df.empty:
        return turns

    grouped = df.groupby("turn_id", sort=True)
    for tid, g in grouped:
        g = g.sort_values("created_at")
        # pregunta del usuario (primera en el turno)
        user_rows = g[g["role"] == "user"]
        if user_rows.empty:
            continue
        q_row = user_rows.iloc[0]
        question = q_row["content"]
        q_time = q_row["created_at"]
        # primera respuesta del asistente (si hay)
        asst_rows = g[g["role"] == "assistant"]
        if not asst_rows.empty:
            a_row = asst_rows.iloc[0]
            answer = truncate(a_row["content"], MAX_ANSWER_CHARS_IN_PROMPT)
            a_time = a_row["created_at"]
            a_model = str(a_row.get("model") or "")
        else:
            answer = ""
            a_time = pd.NaT
            a_model = ""
        turns.append({
            "turn_id": int(tid),
            "question": str(question),
            "question_time": q_time.isoformat().replace("+00:00", "Z") if pd.notna(q_time) else "",
            "answer": str(answer),
            "answer_time": a_time.isoformat().replace("+00:00", "Z") if pd.notna(a_time) else "",
            "answer_model": a_model
        })
    return turns


def cap_turns(turns: List[Dict[str, Any]], max_turns: int = MAX_TURNS_FOR_LLM) -> List[Dict[str, Any]]:
    if len(turns) <= max_turns:
        return turns
    # Estrategia: nos quedamos con los más recientes
    return turns[-max_turns:]


# =========================
# LLM
# =========================

SYSTEM_MSG = (
    "Eres un analista educativo. Dadas interacciones de un estudiante (preguntas, "
    "primera respuesta y fecha), produce un JSON con su perfil de dudas."
)

USER_TEMPLATE = """
Tienes los turnos (ordenados) de un estudiante, cada uno con:
- question_time (ISO8601 UTC)
- question (texto del usuario)
- answer (respuesta del asistente, recortada)
Devuelve SOLO un objeto JSON con:

- "items": lista ordenada (uno por turno) con:
   - "turn_id"
   - "question_time"
   - "category": etiqueta breve (p.ej., "Árboles", "Montículos", "Complejidad", "C/Punteros", "Java/POO", "RAG/LLM", etc.)
   - "subtopic": subtema breve (p.ej., "AVL/Rotaciones", "Heapify vs push", "O(log n)", "NullPointer", etc.)
   - "notes": opcional, breve frase útil (máx 120 caracteres)

- "categories": agregación de dudas con:
   - "category"
   - "subtopic"
   - "count": número de turnos que caen aquí
   - "examples": hasta 2 ejemplos breves de "question"

- "summary": 3–5 frases que describan lagunas y recomendaciones accionables para este estudiante.

NO incluyas nada fuera de JSON. NO repitas todo el contenido. Sé consistente con las etiquetas.

Turnos:
{turns}
""".strip()


def call_llm(client: OpenAI, turns: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    # formatea turnos para prompt (mantenemos igual; el modelo no afecta al análisis)
    lines = []
    for t in turns:
        lines.append(f"- [{t['question_time']}] #{t['turn_id']} Q: {t['question']}")
        if t.get("answer"):
            lines.append(f"  A: {t['answer']}")
    prompt = USER_TEMPLATE.format(turns="\n".join(lines))

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt}
        ]
    )
    text = completion.choices[0].message.content.strip()

    # Limpieza si viene con fences ```
    raw = text
    if raw.startswith("```"):
        raw = raw.strip("`")
        parts = raw.split("\n", 1)
        if len(parts) == 2 and parts[0].lower().startswith("json"):
            raw = parts[1].rsplit("```", 1)[0].strip()
        else:
            raw = raw.split("```", 1)[0].strip()

    # Intento de parseo
    try:
        data = json.loads(raw)
        # Validación mínima
        if not isinstance(data, dict) or "items" not in data or "categories" not in data:
            raise ValueError("Estructura inesperada")
        return data
    except Exception:
        # Fallback minimalista si no es JSON: empaquetamos el texto en summary
        return {"items": [], "categories": [], "summary": text}


# =========================
# Persistencia
# =========================

def write_user_json(out_dir: str, idcv: str, payload: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{idcv}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


# =========================
# Pipeline por fichero
# =========================

def process_file(client: OpenAI, csv_path: str, out_dir: str, model: str) -> Tuple[str, bool, str]:
    """
    Procesa un CSV de un usuario y guarda su JSON con el resumen de dudas + stats de modelo.
    Devuelve (idcv, ok, path_o_motivo).
    """
    idcv = idcv_from_filename(csv_path)
    try:
        df = load_user_dialog(csv_path)
        turns = extract_turns(df)
        if not turns:
            return idcv, False, "sin turnos (no hay preguntas del usuario)"
        turns_cap = cap_turns(turns, MAX_TURNS_FOR_LLM)

        # --- Resumen de dudas con LLM ---
        result = call_llm(client, turns_cap, model=model)

        # --- Estadísticas de modelos (usando SOLO los turnos enviados al LLM) ---
        models_used = [t.get("answer_model", "") for t in turns_cap if t.get("answer_model")]
        models_stats = dict(Counter(models_used))
        turns_model = [{"turn_id": t["turn_id"], "model": t.get("answer_model", "")} for t in turns_cap]

        payload = {
            "idcv": idcv,
            "generated_at": now_iso_utc(),
            "input_stats": {
                "turns_total": len(turns),
                "turns_used": len(turns_cap)
            },
            "models_stats": models_stats,     # <-- agregado
            "turns_model": turns_model,       # <-- agregado
            "items": result.get("items", []),
            "categories": result.get("categories", []),
            "summary": result.get("summary", "")
        }
        out_path = write_user_json(out_dir, idcv, payload)
        return idcv, True, out_path
    except Exception as e:
        return idcv, False, f"error: {e}"


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Procesa CSVs de out_dialogs (por usuario) y genera resúmenes JSON en out_summaries (map) + estadísticas de modelo."
    )
    parser.add_argument("--in", dest="in_dir", default=DEFAULT_IN_DIR, help="Carpeta de entrada de CSVs (uno por idcv)")
    parser.add_argument("--out", dest="out_dir", default=DEFAULT_OUT_DIR, help="Carpeta de salida de JSONs")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo OpenAI (p.ej., gpt-5-mini o gpt-5)")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("❌ Falta OPENAI_API_KEY en .env o entorno.")
    client = OpenAI(api_key=api_key)

    csv_files = sorted(glob.glob(os.path.join(args.in_dir, "*.csv")))
    if not csv_files:
        print(f"No se encontraron CSVs en {args.in_dir}")
        return

    print(f"Procesando {len(csv_files)} archivos de {args.in_dir} → {args.out_dir}")
    ok_count = 0
    for path in csv_files:
        idcv, ok, info = process_file(client, path, args.out_dir, args.model)
        if ok:
            ok_count += 1
            print(f"✓ {idcv} → {info}")
        else:
            print(f"✗ {idcv} → {info}")

    print(f"\n✅ Hecho. JSONs correctos: {ok_count}/{len(csv_files)}")

if __name__ == "__main__":
    main()
