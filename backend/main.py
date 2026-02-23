#!/usr/bin/env python3
"""
Deep Research Agent — Eventos que generan tráfico de internet en Argentina.

Entry point: Asks the user for a target date, runs the LangGraph agent,
and outputs a pandas DataFrame table.

Usage:
    uv run python main.py
"""

import json
import logging
import sys
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from tabulate import tabulate

from graph import build_graph

# ─────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def ask_date() -> str:
    """Prompt user for target date."""
    print("\n" + "=" * 60)
    print("🔍 DEEP RESEARCH — Tráfico de Internet en Argentina")
    print("=" * 60)
    print("\nIngresá la fecha que querés investigar.")
    print("Formato: YYYY-MM-DD (ej: 2026-02-23)")
    print("O presioná Enter para usar la fecha de mañana.\n")

    user_input = input("📅 Fecha: ").strip()

    if not user_input:
        tomorrow = datetime.now() + timedelta(days=1)
        date_str = tomorrow.strftime("%Y-%m-%d")
        print(f"   → Usando fecha de mañana: {date_str}")
        return date_str

    # Validate format
    try:
        datetime.strptime(user_input, "%Y-%m-%d")
        return user_input
    except ValueError:
        print("❌ Formato inválido. Usando fecha de mañana.")
        tomorrow = datetime.now() + timedelta(days=1)
        return tomorrow.strftime("%Y-%m-%d")


def display_results(report_json: str, target_date: str):
    """Parse JSON report and display as pandas table."""
    print("\n" + "=" * 60)
    print(f"📊 EVENTOS QUE GENERAN TRÁFICO — {target_date}")
    print("=" * 60 + "\n")

    try:
        events = json.loads(report_json)
    except json.JSONDecodeError:
        print("⚠️  No se pudo parsear el reporte. Resultado crudo:")
        print(report_json)
        return None

    if not events:
        print("ℹ️  No se encontraron eventos relevantes para esta fecha.")
        return pd.DataFrame()

    df = pd.DataFrame(events)

    # Reorder columns if present
    desired_cols = [
        "evento", "categoria", "fecha", "hora_argentina",
        "descripcion", "impacto_estimado", "fuente"
    ]
    cols = [c for c in desired_cols if c in df.columns]
    df = df[cols]

    # Rename for display
    col_names = {
        "evento": "Evento",
        "categoria": "Categoría",
        "fecha": "Fecha",
        "hora_argentina": "Hora (ARG)",
        "descripcion": "Descripción",
        "impacto_estimado": "Impacto",
        "fuente": "Fuente",
    }
    df = df.rename(columns=col_names)

    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False, maxcolwidths=40))
    print(f"\n📈 Total: {len(df)} eventos encontrados\n")

    return df


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

def main():
    """Run the deep research agent."""
    target_date = ask_date()

    print(f"\n🚀 Iniciando investigación para: {target_date}")
    print("   Esto puede tomar unos minutos...\n")

    # Build and run the graph
    app = build_graph()

    initial_state = {
        "target_date": target_date,
        "search_plan": None,
        "raw_events": [],
        "filtered_events": None,
        "final_report": None,
    }

    # Stream to show progress
    final_state = None
    for event in app.stream(initial_state, stream_mode="updates"):
        for node_name, update in event.items():
            if node_name == "plan_research":
                plan = update.get("search_plan")
                if plan:
                    print(f"\n📋 Plan: {len(plan.queries)} queries generadas")
                    for sq in plan.queries:
                        print(f"   [{sq.category}] {sq.query}")
            elif node_name == "research_category":
                events = update.get("raw_events", [])
                if events:
                    cat = events[0].get("categoria", "?") if events else "?"
                    print(f"🔍 Researcher [{cat}]: {len(events)} eventos encontrados")
                else:
                    print(f"🔍 Researcher: sin eventos para esta búsqueda")
            elif node_name == "aggregate_results":
                raw = update.get("raw_events", {})
                if isinstance(raw, dict) and raw.get("type") == "override":
                    count = len(raw.get("value", []))
                else:
                    count = len(raw) if isinstance(raw, list) else 0
                print(f"\n📦 Agregados: {count} eventos únicos")
            elif node_name == "filter_argentina":
                filtered = update.get("filtered_events", [])
                print(f"🇦🇷 Filtrados: {len(filtered)} eventos relevantes para Argentina")
            elif node_name == "generate_report":
                print("📄 Reporte final generado")
                final_state = update

    # Get the report from streamed results
    report = "[]"
    if final_state and "final_report" in final_state:
        report = final_state["final_report"]

    # Display as pandas table
    df = display_results(report, target_date)

    # Optionally save to CSV
    if df is not None and not df.empty:
        csv_path = f"eventos_{target_date}.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 Tabla guardada en: {csv_path}")

    return df


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Investigación cancelada.")
        sys.exit(0)
