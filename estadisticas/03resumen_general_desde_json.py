#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import argparse
from collections import defaultdict, Counter
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

from dotenv import load_dotenv

# OpenAI SDK >=1.0
try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("Falta 'openai'. Instala: pip install openai python-dotenv")

# =========================
# Config
# =========================

load_dotenv()  # lee .env (OPENAI_API_KEY)

DEFAULT_IN_DIR = "estadisticas/out_summaries"  # entrada (sin tilde, como tenías)
OUT_BASE_DIR = "estadisticas/out_global"       # <-- carpeta solicitada (con tilde)

DEFAULT_OUT_CSV_CATEGORIES = os.path.join(OUT_BASE_DIR, "global_categories.csv")
DEFAULT_OUT_JSON_AGG       = os.path.join(OUT_BASE_DIR, "global_categories.json")
DEFAULT_OUT_MD_REPORT      = os.path.join(OUT_BASE_DIR, "global_report.md")
# nuevos CSV para modelos
DEFAULT_OUT_CSV_MODELS_GLOBAL   = os.path.join(OUT_BASE_DIR, "global_models.csv")
DEFAULT_OUT_CSV_MODELS_BY_USER  = os.path.join(OUT_BASE_DIR, "global_models_by_user.csv")

DEFAULT_MODEL = "gpt-5-mini"

TOP_N_FOR_PROMPT = 200   # límite de pares (category, subtopic) que enviaremos al LLM
MAX_EXAMPLES_PER_PAIR = 2


# =========================
# Utilidades
# =========================

def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def norm(s: str) -> str:
    return (s or "").strip()

def key_pair(cat: str, sub: str) -> Tuple[str, str]:
    return (norm(cat), norm(sub))

def pick_examples(ex_list: List[str], k: int) -> List[str]:
    # quita duplicados preservando orden
    seen = set()
    out = []
    for e in ex_list:
        e = (e or "").strip()
        if not e or e in seen:
            continue
        seen.add(e)
        out.append(e)
        if len(out) >= k:
            break
    return out


# =========================
# Carga
# =========================

def load_user_jsons(in_dir: str) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(in_dir, "*.json")))
    payloads = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payloads.append(json.load(f))
        except Exception as e:
            print(f"⚠️  No se pudo leer {path}: {e}")
    return payloads


# =========================
# Agregación
# =========================

def aggregate(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Devuelve estructura agregada:
    {
      "generated_at": "...Z",
      "users_processed": N,
      "total_items": M,
      "pairs": [
         {"category": "...", "subtopic": "...", "total_count": X, "examples": ["..",".."]}
      ],
      "models_global": {"gpt-4o-mini": 12, "gpt-5-mini": 34, ...},
      "models_by_user": [{"user_id":"...","model":"gpt-5-mini","count":n}, ...]
    }
    """
    # --- Agregado de categorías/subtemas ---
    freq_pairs: Counter = Counter()
    examples: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    total_items = 0

    # --- Agregado de modelos ---
    models_global: Counter = Counter()
    # acumulador por usuario
    per_user_counts: Dict[str, Counter] = defaultdict(Counter)

    for p in payloads:
        # id usuario
        user_id = str(p.get("idcv") or p.get("user_id") or "unknown")

        # 1) categorías/subtemas
        cats = p.get("categories", [])
        for c in cats:
            cat = norm(c.get("category"))
            sub = norm(c.get("subtopic"))
            cnt = int(c.get("count") or 0)
            if not cat and not sub:
                continue
            k = key_pair(cat, sub)
            freq_pairs[k] += cnt
            # agrega ejemplos (limitaremos más tarde)
            exs = c.get("examples") or []
            for e in exs:
                if e and len(examples[k]) < 50:
                    examples[k].append(str(e))
            total_items += cnt

        # 2) modelos (preferimos models_stats; si no, inferimos de turns_model)
        user_models_stats = p.get("models_stats") or {}
        if isinstance(user_models_stats, dict) and user_models_stats:
            # suma directa
            models_global.update({m: int(n or 0) for m, n in user_models_stats.items()})
            per_user_counts[user_id].update({m: int(n or 0) for m, n in user_models_stats.items()})
        else:
            # fallback: contar desde turns_model (lista de dicts con turn_id y model)
            turns_model = p.get("turns_model") or []
            tmp = Counter([tm.get("model") or "" for tm in turns_model if (tm.get("model") or "").strip()])
            if tmp:
                models_global.update(tmp)
                per_user_counts[user_id].update(tmp)

    # Formato de salida para pares
    pairs = []
    for (cat, sub), count in freq_pairs.most_common():
        pairs.append({
            "category": cat,
            "subtopic": sub,
            "total_count": int(count),
            "examples": pick_examples(examples.get((cat, sub), []), MAX_EXAMPLES_PER_PAIR)
        })

    # Formato de salida para modelos
    models_global_dict = dict(sorted(models_global.items(), key=lambda x: (-x[1], x[0])))
    models_by_user_rows = []
    for uid, counter in sorted(per_user_counts.items(), key=lambda x: x[0]):
        for model, cnt in counter.most_common():
            models_by_user_rows.append({"user_id": uid, "model": model, "count": int(cnt)})

    return {
        "generated_at": now_iso_utc(),
        "users_processed": len(payloads),
        "total_items": int(total_items),
        "pairs": pairs,
        "models_global": models_global_dict,
        "models_by_user": models_by_user_rows
    }


# =========================
# Persistencia
# =========================

def ensure_outdir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def write_csv_categories(pairs: List[Dict[str, Any]], out_csv: str) -> None:
    import csv
    ensure_outdir(out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "category", "subtopic", "total_count", "example_1", "example_2"])
        for i, p in enumerate(pairs, start=1):
            ex = p.get("examples", [])
            w.writerow([
                i,
                p.get("category",""),
                p.get("subtopic",""),
                p.get("total_count",0),
                ex[0] if len(ex)>0 else "",
                ex[1] if len(ex)>1 else ""
            ])

def write_csv_models_global(models_global: Dict[str, int], out_csv: str) -> None:
    import csv
    ensure_outdir(out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "count"])
        for model, cnt in models_global.items():
            w.writerow([model, int(cnt)])

def write_csv_models_by_user(models_by_user: List[Dict[str, Any]], out_csv: str) -> None:
    import csv
    ensure_outdir(out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "model", "count"])
        for row in models_by_user:
            w.writerow([row.get("user_id",""), row.get("model",""), int(row.get("count",0))])

def write_json(aggregate_obj: Dict[str, Any], out_json: str) -> None:
    ensure_outdir(out_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(aggregate_obj, f, ensure_ascii=False, indent=2)

def write_markdown(md: str, out_md: str) -> None:
    ensure_outdir(out_md)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)


# =========================
# LLM resumen global (categorías)
# =========================

SYSTEM_MSG = (
    "Eres un analista educativo senior. Te paso un agregado global de dudas "
    "de múltiples estudiantes (categorías, subtemas, frecuencias y ejemplos). "
    "Redacta un informe ejecutivo claro y accionable."
)

USER_TEMPLATE = """
Estos son los pares (category, subtopic) más frecuentes con su conteo global.
Incluye también 0–2 ejemplos reales por par.

Devuélveme un **informe en Markdown** con:
1) Top categorías y subtemas (con % aproximado sobre el total).
2) Patrones y causas probables.
3) Recomendaciones accionables priorizadas (Alta/Media/Baja).
4) 7 FAQs propuestas con respuesta de una línea cada una.
5) Métricas a monitorizar.

Datos (top {top_n}):
{lines}

Total items (suma de counts): {total_items}
Usuarios procesados: {users}
""".strip()

def build_prompt_lines(pairs: List[Dict[str, Any]], top_n: int, total_items: int, users: int) -> str:
    subset = pairs[:top_n]
    lines = []
    for p in subset:
        cat = p.get("category", "")
        sub = p.get("subtopic", "")
        cnt = int(p.get("total_count", 0))
        exs = p.get("examples", [])
        line = f"- ({cnt}) {cat} :: {sub}"
        if exs:
            line += f" | Ej: { '; '.join(exs[:MAX_EXAMPLES_PER_PAIR]) }"
        lines.append(line)
    return "\n".join(lines)

def call_llm_report(pairs: List[Dict[str, Any]], total_items: int, users: int, model: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("❌ Falta OPENAI_API_KEY en .env o entorno.")
    client = OpenAI(api_key=api_key)

    lines = build_prompt_lines(pairs, TOP_N_FOR_PROMPT, total_items, users)
    user_msg = USER_TEMPLATE.format(
        top_n=min(TOP_N_FOR_PROMPT, len(pairs)),
        lines=lines,
        total_items=total_items,
        users=users
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg}
        ]
    )
    return completion.choices[0].message.content.strip()


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Lee JSONs de out_summaries y genera un resumen global de dudas y uso de modelos (CSV/JSON/MD)."
    )
    parser.add_argument("--in", dest="in_dir", default=DEFAULT_IN_DIR, help="Carpeta con JSONs por usuario")
    parser.add_argument("--csv-cats", dest="out_csv_categories",
                        default=DEFAULT_OUT_CSV_CATEGORIES, help="CSV categorías global")
    parser.add_argument("--csv-models", dest="out_csv_models",
                        default=DEFAULT_OUT_CSV_MODELS_GLOBAL, help="CSV modelos global")
    parser.add_argument("--csv-models-user", dest="out_csv_models_by_user",
                        default=DEFAULT_OUT_CSV_MODELS_BY_USER, help="CSV modelos por usuario")
    parser.add_argument("--json", dest="out_json",
                        default=DEFAULT_OUT_JSON_AGG, help="Ruta JSON agregado de salida")
    parser.add_argument("--md", dest="out_md",
                        default=DEFAULT_OUT_MD_REPORT, help="Ruta reporte Markdown (LLM + modelos)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo (gpt-5-mini | gpt-5) para el informe LLM de categorías")
    args = parser.parse_args()

    payloads = load_user_jsons(args.in_dir)
    if not payloads:
        print(f"No se encontraron JSONs en {args.in_dir}")
        return

    agg = aggregate(payloads)
    pairs = agg["pairs"]
    total_items = agg["total_items"]
    users = agg["users_processed"]
    models_global = agg.get("models_global", {})
    models_by_user = agg.get("models_by_user", [])

    # Persistimos CSVs
    write_csv_categories(pairs, args.out_csv_categories)
    write_csv_models_global(models_global, args.out_csv_models)
    write_csv_models_by_user(models_by_user, args.out_csv_models_by_user)

    # Persistimos JSON completo (incluye modelos)
    write_json(agg, args.out_json)

    # LLM para informe global (categorías)
    report_md = call_llm_report(pairs, total_items, users, model=args.model)

    # Añadimos sección de modelos (clara y determinista) antes o después del informe
    models_section = ["# Uso de modelos (agregado)",
                      "",
                      "| Modelo | Respuestas |",
                      "|---|---|"]
    for m, cnt in models_global.items():
        models_section.append(f"| {m or '(vacío)'} | {int(cnt)} |")
    models_section.append("")  # newline

    final_md = "\n".join(models_section) + "\n\n" + report_md
    write_markdown(final_md, args.out_md)

    print(f"✅ CSV categorías: {os.path.abspath(args.out_csv_categories)}")
    print(f"✅ CSV modelos:    {os.path.abspath(args.out_csv_models)}")
    print(f"✅ CSV modelos/u.: {os.path.abspath(args.out_csv_models_by_user)}")
    print(f"✅ JSON:           {os.path.abspath(args.out_json)}")
    print(f"✅ MD:             {os.path.abspath(args.out_md)}")

if __name__ == "__main__":
    main()
