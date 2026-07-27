"""
Extraction des événements bruts depuis l'API Opendatasoft (dataset OpenAgenda),
filtrés par périmètre géographique (Bordeaux Métropole) et fenêtre temporelle
(< DAYS_HISTORY jours), tel que défini dans src/config.py.

Usage :
    # Mode diagnostic : affiche les champs réels renvoyés par l'API, sans
    # rien filtrer ni sauvegarder. À lancer en premier pour valider les noms
    # de champs définis dans config.py avant une extraction complète.
    python -m src.fetch_raw_data --discover

    # Extraction complète (paginée), sauvegarde dans data/raw/
    python -m src.fetch_raw_data
"""

import argparse
import json
import sys
import time
from datetime import date, timedelta

import requests

from src import config


def build_where_clause(commune: str) -> str:
    """Construit la clause ODSQL `where` pour UNE commune : date récente + ville exacte.

    Pourquoi une commune à la fois et pas un `in (...)` sur les 28 d'un coup :
    l'API plafonne `offset + limit` à 10 000 sur l'ensemble des résultats d'UNE
    requête, tous filtres confondus. Bordeaux Métropole dépasse ce seuil
    (~10 700 événements). En interrogeant commune par commune, chaque requête
    a son propre total (bien en dessous de 10 000 pour n'importe quelle commune
    prise isolément) et sa propre pagination qui repart de zéro.
    """
    threshold = (date.today() - timedelta(days=config.DAYS_HISTORY)).isoformat()
    return (
        f"{config.FIELD_FIRSTDATE_BEGIN} >= date'{threshold}' "
        f'AND {config.FIELD_LOCATION_CITY} = "{commune}"'
    )


def fetch_page(offset: int, where: str | None, limit: int) -> dict:
    """Récupère une page de résultats. Lève une exception explicite en cas d'erreur HTTP."""
    params = {"limit": limit, "offset": offset}
    if where:
        params["where"] = where

    last_error = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            response = requests.get(
                config.API_BASE_URL,
                params=params,
                timeout=config.REQUEST_TIMEOUT_SECONDS,
                headers={"User-Agent": "puls-events-rag-poc/1.0"},
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"L'API a répondu {response.status_code}.\n"
                    f"URL appelée : {response.url}\n"
                    f"Corps de la réponse : {response.text[:1000]}\n\n"
                    "Si l'erreur mentionne un champ inconnu, c'est probablement un nom de "
                    "champ à corriger dans src/config.py (voir python -m src.fetch_raw_data --discover)."
                )
            return response.json()
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt < config.MAX_RETRIES:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Échec après {config.MAX_RETRIES} tentatives : {last_error}")


def discover_schema() -> None:
    """Récupère 1 enregistrement SANS filtre et affiche tous les champs bruts renvoyés."""
    print("Mode diagnostic : récupération d'un enregistrement sans filtre...\n")
    data = fetch_page(offset=0, where=None, limit=1)
    total = data.get("total_count", "inconnu")
    print(f"total_count (dataset complet, sans filtre) : {total}\n")

    results = data.get("results", [])
    if not results:
        print("Aucun enregistrement renvoyé — impossible d'inspecter le schéma.")
        sys.exit(1)

    record = results[0]
    print(f"Champs renvoyés par l'API ({len(record)} champs) :\n")
    for field_name, value in sorted(record.items()):
        preview = str(value)
        if len(preview) > 80:
            preview = preview[:80] + "..."
        print(f"  {field_name:30s} = {preview}")

    print(
        "\nCompare ces noms de champs avec les constantes FIELD_* dans src/config.py "
        "et corrige-les si besoin avant de lancer une extraction complète."
    )


def fetch_commune(commune: str) -> list[dict]:
    """Récupère TOUS les événements d'UNE commune (pagination propre à cette commune)."""
    where = build_where_clause(commune)
    commune_results: list[dict] = []
    offset = 0
    total_count = 0

    for _ in range(1, config.MAX_PAGES + 1):
        data = fetch_page(offset=offset, where=where, limit=config.PAGE_SIZE)
        total_count = data.get("total_count", 0)
        page_results = data.get("results", [])
        commune_results.extend(page_results)

        offset += config.PAGE_SIZE
        if offset >= total_count or not page_results:
            break
        time.sleep(config.REQUEST_DELAY_SECONDS)
    else:
        print(f"  [!] Limite de {config.MAX_PAGES} pages atteinte pour {commune} "
              f"({len(commune_results)}/{total_count}) : extraction incomplète pour "
              f"cette commune. Augmenter MAX_PAGES dans config.py si besoin.")

    return commune_results


def fetch_all() -> tuple[list[dict], dict]:
    """Récupère tous les événements du périmètre, commune par commune."""
    all_results: list[dict] = []
    per_commune_counts: dict[str, int] = {}

    for i, commune in enumerate(config.BORDEAUX_METROPOLE_COMMUNES, start=1):
        commune_results = fetch_commune(commune)
        per_commune_counts[commune] = len(commune_results)
        all_results.extend(commune_results)
        print(f"[{i:2d}/{len(config.BORDEAUX_METROPOLE_COMMUNES)}] {commune:30s} : "
              f"{len(commune_results)} événements (total cumulé : {len(all_results)})")
        time.sleep(config.REQUEST_DELAY_SECONDS)

    meta = {
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_id": config.DATASET_ID,
        "strategie_pagination": "une requete par commune (limite API : offset+limit <= 10000 par requete)",
        "records_fetched": len(all_results),
        "per_commune_counts": per_commune_counts,
        "communes": config.BORDEAUX_METROPOLE_COMMUNES,
        "days_history": config.DAYS_HISTORY,
    }
    return all_results, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discover", action="store_true",
        help="Affiche les champs bruts renvoyés par l'API sans rien filtrer ni sauvegarder.",
    )
    args = parser.parse_args()

    if args.discover:
        discover_schema()
        return

    results, meta = fetch_all()

    if not results:
        print(
            "\n[!] Aucun événement trouvé pour ce périmètre/cette période. "
            "Vérifie la clause where ci-dessus, ou élargis le périmètre géographique "
            "dans src/config.py."
        )

    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.RAW_DATA_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config.RAW_META_FILE.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{len(results)} événements sauvegardés dans {config.RAW_DATA_FILE}")
    print(f"Métadonnées de l'extraction dans {config.RAW_META_FILE}")


if __name__ == "__main__":
    main()
