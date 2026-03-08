#!/usr/bin/env python3
"""
Agent Viajes — Rutas turísticas baratas con combinaciones.

Entry point: Asks the user for travel details, runs the LangGraph agent,
and outputs a pandas DataFrame table with the cheapest routes.

Usage:
    cd agents/agent_viajes
    uv run python main.py
"""

import json
import logging
import sys

from dotenv import load_dotenv
from pathlib import Path

# Load .env from root project directory
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")

import pandas as pd
from tabulate import tabulate

from graph import build_graph

# ─────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def ask_travel_details() -> dict:
    """Prompt user for travel details."""
    print("\n" + "=" * 60)
    print("✈️  AGENT VIAJES — Rutas Turísticas Baratas")
    print("=" * 60)

    print("\nIngresá los detalles de tu viaje:\n")

    origin = input("📍 Ciudad de origen [Buenos Aires]: ").strip()
    if not origin:
        origin = "Buenos Aires"

    destination = input("🎯 Destino (ciudad o evento, ej: 'Mundial 2026 USA'): ").strip()
    if not destination:
        destination = "Mundial 2026 USA"
        print(f"   → Usando destino: {destination}")

    travel_dates = input("📅 Fechas (ej: '2026-06-15 al 2026-07-05'): ").strip()
    if not travel_dates:
        travel_dates = "2026-06-15 al 2026-07-05"
        print(f"   → Usando fechas: {travel_dates}")

    flex_str = input("🔄 Flexibilidad en días [3]: ").strip()
    flexibility = int(flex_str) if flex_str.isdigit() else 3

    budget_str = input("💰 Presupuesto máximo USD (Enter = sin límite): ").strip()
    budget = float(budget_str) if budget_str else None

    return {
        "origin": origin,
        "destination": destination,
        "travel_dates": travel_dates,
        "flexibility_days": flexibility,
        "budget_max_usd": budget,
    }


def display_results(report_json: str, details: dict):
    """Parse JSON report and display as pandas table."""
    print("\n" + "=" * 60)
    print(f"🏆 RUTAS MÁS BARATAS — {details['origin']} → {details['destination']}")
    print("=" * 60 + "\n")

    try:
        routes = json.loads(report_json)
    except json.JSONDecodeError:
        print("⚠️  No se pudo parsear el reporte. Resultado crudo:")
        print(report_json)
        return None

    if not routes:
        print("ℹ️  No se encontraron rutas para estos parámetros.")
        return pd.DataFrame()

    df = pd.DataFrame(routes)

    # Reorder columns if present
    desired_cols = [
        "ranking", "ruta", "aerolineas", "precio_usd",
        "duracion_total", "escalas", "tipo", "notas", "fuente"
    ]
    cols = [c for c in desired_cols if c in df.columns]
    df = df[cols]

    # Rename for display
    col_names = {
        "ranking": "#",
        "ruta": "Ruta",
        "aerolineas": "Aerolíneas",
        "precio_usd": "Precio (USD)",
        "duracion_total": "Duración",
        "escalas": "Escalas",
        "tipo": "Tipo",
        "notas": "Notas",
        "fuente": "Fuente",
    }
    df = df.rename(columns=col_names)

    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False, maxcolwidths=35))
    print(f"\n📈 Total: {len(df)} rutas encontradas\n")

    return df


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

def main():
    """Run the travel research agent."""
    details = ask_travel_details()

    print(f"\n🚀 Buscando rutas baratas: {details['origin']} → {details['destination']}")
    print("   Esto puede tomar unos minutos...\n")

    # Build and run the graph
    app = build_graph()

    initial_state = {
        "origin": details["origin"],
        "destination": details["destination"],
        "travel_dates": details["travel_dates"],
        "flexibility_days": details["flexibility_days"],
        "budget_max_usd": details["budget_max_usd"],
        "search_plan": None,
        "raw_routes": [],
        "ranked_routes": None,
        "final_itinerary": None,
    }

    # Stream to show progress
    final_state = None
    for event in app.stream(initial_state, stream_mode="updates"):
        for node_name, update in event.items():
            if node_name == "plan_search":
                plan = update.get("search_plan")
                if plan:
                    print(f"\n📋 Plan: {len(plan.queries)} queries generadas")
                    for fq in plan.queries:
                        print(f"   [{fq.search_type}] {fq.query}")
            elif node_name == "search_flights":
                routes = update.get("raw_routes", [])
                if routes:
                    tipo = routes[0].get("tipo", "?") if routes else "?"
                    print(f"🔍 Searcher [{tipo}]: {len(routes)} rutas encontradas")
                else:
                    print(f"🔍 Searcher: sin rutas para esta búsqueda")
            elif node_name == "aggregate_routes":
                raw = update.get("raw_routes", {})
                if isinstance(raw, dict) and raw.get("type") == "override":
                    count = len(raw.get("value", []))
                else:
                    count = len(raw) if isinstance(raw, list) else 0
                print(f"\n📦 Agregadas: {count} rutas únicas")
            elif node_name == "rank_and_optimize":
                ranked = update.get("ranked_routes", [])
                print(f"🏆 Ranqueadas: {len(ranked)} rutas por precio/conveniencia")
            elif node_name == "generate_itinerary":
                print("📄 Itinerario final generado")
                final_state = update

    # Get the report from streamed results
    report = "[]"
    if final_state and "final_itinerary" in final_state:
        report = final_state["final_itinerary"]

    # Display as pandas table
    df = display_results(report, details)

    # Optionally save to CSV
    if df is not None and not df.empty:
        dest_clean = details["destination"].replace(" ", "_").lower()[:20]
        csv_path = f"rutas_{dest_clean}.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 Tabla guardada en: {csv_path}")

    return df


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Búsqueda cancelada.")
        sys.exit(0)
