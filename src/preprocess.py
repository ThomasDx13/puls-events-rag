"""
Nettoyage, re-validation et structuration des événements bruts en un jeu de
données propre, prêt à être indexé (étape 3).

Ce script NE FAIT PAS confiance au filtrage côté API : il revalide
systématiquement la date et la ville de chaque événement, pour garantir que
le fichier de sortie respecte les règles métier même si le filtre serveur
a été mal construit (nom de champ erroné, etc.). C'est cette revalidation
que les tests unitaires (étape suivante) viendront vérifier.

Usage :
    python -m src.preprocess
"""

import json
import json
from datetime import date, timedelta

from bs4 import BeautifulSoup

from src import config


def load_raw_events() -> list[dict]:
    if not config.RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"{config.RAW_DATA_FILE} introuvable. "
            "Lance d'abord `python -m src.fetch_raw_data`."
        )
    return json.loads(config.RAW_DATA_FILE.read_text(encoding="utf-8"))


def _clean_text(value) -> str:
    if not value or not isinstance(value, str):
        return ""
    return " ".join(value.split())  # normalise les espaces/retours à la ligne


def _is_recent_enough(firstdate_begin: str | None) -> bool:
    if not firstdate_begin:
        return False
    try:
        event_date = date.fromisoformat(firstdate_begin[:10])
    except ValueError:
        return False
    threshold = date.today() - timedelta(days=config.DAYS_HISTORY)
    return event_date >= threshold


def _parse_timings(raw_value) -> list:
    """Le champ `timings` renvoyé par l'API est une CHAÎNE contenant du JSON
    (ex: '[{"begin": "2022-11-16T15:00:00+01:00", "end": "..."}]'), pas une
    vraie liste Python — vérifié le 16/07/2026 sur un enregistrement réel.
    On la parse ici. Si le parsing échoue (champ absent, mal formé, ou déjà
    une liste dans un format futur de l'API), on ne fait PAS planter le
    traitement de l'événement pour autant : `timings` est une information
    secondaire (les dates principales sont dans firstdate_begin/lastdate_end),
    donc on retombe sur une liste vide plutôt que de rejeter l'événement.
    """
    if isinstance(raw_value, list):
        return raw_value
    if not raw_value or not isinstance(raw_value, str):
        return []
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _clean_html(raw_html) -> str:
    """Nettoie le HTML de longdescription_fr (ex: '<p><strong>...</strong></p>')
    en texte brut lisible. On utilise BeautifulSoup plutôt qu'une regex
    maison car le HTML réel imbrique plusieurs balises (<p>, <strong>, <br>,
    <em>...) — une regex simpliste (`re.sub(r"<.*?>", "", texte)`) fonctionne
    sur des cas simples mais devient vite fragile sur du HTML réel.

    `separator=" "` évite qu'une balise fermante collée à la suivante ne
    fusionne deux phrases sans espace (ex: deux <p> à la suite donneraient
    "...pain.Une programmation..." sans le séparateur).
    """
    if not raw_html or not isinstance(raw_html, str):
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return _clean_text(soup.get_text(separator=" "))


def _parse_registration(raw_value) -> list[dict]:
    """Le champ `registration` est une chaîne JSON représentant une liste de
    moyens d'inscription/contact (ex: '[{"type": "link", "value": "..."},
    {"type": "email", "value": "..."}]'). Même logique défensive que
    _parse_timings : on ne fait jamais planter le traitement d'un événement
    pour un champ secondaire mal formé.
    """
    if not raw_value or not isinstance(raw_value, str):
        return []
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _is_in_perimeter(city: str | None) -> bool:
    if not city:
        return False
    return city.strip() in config.BORDEAUX_METROPOLE_COMMUNES


def clean_and_structure(raw_events: list[dict]) -> tuple[list[dict], dict]:
    """Filtre + nettoie + restructure. Retourne (événements propres, rapport de rejet)."""
    seen_uids = set()
    clean_events = []
    rejected = {
        "doublon": 0,
        "titre_manquant": 0,
        "description_manquante": 0,
        "date_hors_perimetre": 0,
        "ville_hors_perimetre": 0,
    }
    timings_non_parsables = 0

    for raw in raw_events:
        uid = raw.get(config.FIELD_UID)

        if uid in seen_uids:
            rejected["doublon"] += 1
            continue

        title = _clean_text(raw.get(config.FIELD_TITLE))
        if not title:
            rejected["titre_manquant"] += 1
            continue

        description = _clean_text(raw.get(config.FIELD_DESCRIPTION)) or _clean_text(
            raw.get(config.FIELD_LONGDESCRIPTION)
        )
        if not description:
            rejected["description_manquante"] += 1
            continue

        firstdate_begin = raw.get(config.FIELD_FIRSTDATE_BEGIN)
        if not _is_recent_enough(firstdate_begin):
            rejected["date_hors_perimetre"] += 1
            continue

        city = raw.get(config.FIELD_LOCATION_CITY)
        if not _is_in_perimeter(city):
            rejected["ville_hors_perimetre"] += 1
            continue

        seen_uids.add(uid)

        coordinates = raw.get(config.FIELD_LOCATION_COORDINATES) or {}
        parsed_timings = _parse_timings(raw.get(config.FIELD_TIMINGS))
        if raw.get(config.FIELD_TIMINGS) and not parsed_timings:
            timings_non_parsables += 1

        clean_events.append({
            "uid": uid,
            "title": title,
            "description": description,
            "long_description": _clean_html(raw.get(config.FIELD_LONGDESCRIPTION)),
            "conditions": _clean_text(raw.get(config.FIELD_CONDITIONS)),
            "registration": _parse_registration(raw.get(config.FIELD_REGISTRATION)),
            "date_start": firstdate_begin,
            "date_end": raw.get(config.FIELD_LASTDATE_END) or raw.get(config.FIELD_FIRSTDATE_END),
            "timings": parsed_timings,
            "keywords": raw.get(config.FIELD_KEYWORDS) or [],
            "location": {
                "name": _clean_text(raw.get(config.FIELD_LOCATION_NAME)),
                "address": _clean_text(raw.get(config.FIELD_LOCATION_ADDRESS)),
                "city": city,
                "postal_code": raw.get(config.FIELD_LOCATION_POSTALCODE),
                "latitude": coordinates.get("lat") if isinstance(coordinates, dict) else None,
                "longitude": coordinates.get("lon") if isinstance(coordinates, dict) else None,
            },
            "url": raw.get(config.FIELD_CANONICAL_URL),
            "source_agenda": raw.get(config.FIELD_ORIGINAGENDA_TITLE),
        })

    report = {
        "total_brut": len(raw_events),
        "total_propre": len(clean_events),
        "rejetes": rejected,
        "timings_non_parsables": timings_non_parsables,
    }
    return clean_events, report


def main() -> None:
    raw_events = load_raw_events()
    print(f"{len(raw_events)} événements bruts chargés depuis {config.RAW_DATA_FILE}\n")

    clean_events, report = clean_and_structure(raw_events)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DATA_FILE.write_text(
        json.dumps(clean_events, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Rapport de nettoyage :")
    print(f"  Total brut     : {report['total_brut']}")
    for reason, count in report["rejetes"].items():
        print(f"  Rejetés ({reason}) : {count}")
    print(f"  Total propre   : {report['total_propre']}")
    print(f"  (info) timings non parsables (non bloquant) : {report['timings_non_parsables']}")
    print(f"\nJeu de données structuré sauvegardé dans {config.PROCESSED_DATA_FILE}")


if __name__ == "__main__":
    main()
