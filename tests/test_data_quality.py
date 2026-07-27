"""
Vérifie que les VRAIES données indexées (data/processed/chunks.json,
produites par le pipeline complet fetch_raw_data -> preprocess -> vectorize)
respectent bien les règles métier : uniquement des événements de moins d'un
an, dans le périmètre de Bordeaux Métropole. C'est le test qui correspond
littéralement à la consigne du projet ("tester que les données intégrées
dans la base vectorielle correspondent bien à des évènements de moins d'un
an dans la région géographique sélectionnée").

Contrairement à test_preprocess.py, ce test DÉPEND de données déjà générées
— il faut avoir lancé tout le pipeline au moins une fois avant de pouvoir
l'exécuter utilement (sinon il est ignoré avec un message clair plutôt que
de planter).

Point important, à avoir en tête en le relançant plus tard : la comparaison
de date se fait par rapport à AUJOURD'HUI (au moment où le test tourne), pas
par rapport à la date de l'extraction. Si tu relances ce test longtemps
après avoir généré chunks.json sans relancer le pipeline, un événement
valide au moment de l'extraction peut désormais dépasser la limite d'un an
— c'est volontaire : ça signale un index périmé plutôt que de valider
silencieusement une donnée qui ne respecte plus la règle actuelle.

Usage :
    pytest tests/test_data_quality.py -v
"""

import json
from datetime import date, timedelta

import pytest

from src import config


def _load_chunks() -> list[dict]:
    if not config.CHUNKS_FILE.exists():
        pytest.skip(
            f"{config.CHUNKS_FILE} introuvable — lance d'abord tout le pipeline "
            "(python -m src.fetch_raw_data, src.preprocess, src.vectorize) "
            "avant ce test."
        )
    return json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def chunks():
    return _load_chunks()


def test_chunks_json_non_vide(chunks):
    assert len(chunks) > 0, "chunks.json est vide"


def test_tous_les_chunks_ont_une_date_dans_la_fenetre(chunks):
    seuil = date.today() - timedelta(days=config.DAYS_HISTORY)
    violations = []

    for chunk in chunks:
        date_start = chunk["metadata"].get("date_start")
        try:
            event_date = date.fromisoformat(date_start[:10])
        except (ValueError, TypeError):
            violations.append((chunk["metadata"].get("uid"), date_start, "date illisible"))
            continue
        if event_date < seuil:
            violations.append((chunk["metadata"].get("uid"), date_start, "trop ancien"))

    assert not violations, (
        f"{len(violations)} chunk(s) avec une date hors fenêtre "
        f"(seuil : {seuil.isoformat()}) : {violations[:10]}"
        f"{' ... (liste tronquée)' if len(violations) > 10 else ''}"
    )


def test_tous_les_chunks_sont_dans_le_perimetre(chunks):
    violations = [
        (c["metadata"].get("uid"), c["metadata"].get("city"))
        for c in chunks
        if c["metadata"].get("city") not in config.BORDEAUX_METROPOLE_COMMUNES
    ]

    assert not violations, (
        f"{len(violations)} chunk(s) hors périmètre (Bordeaux Métropole) : "
        f"{violations[:10]}{' ... (liste tronquée)' if len(violations) > 10 else ''}"
    )


def test_chaque_chunk_a_un_uid(chunks):
    sans_uid = [i for i, c in enumerate(chunks) if not c["metadata"].get("uid")]
    assert not sans_uid, f"{len(sans_uid)} chunk(s) sans uid (index : {sans_uid[:10]})"
