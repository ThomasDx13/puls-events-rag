"""
Tests unitaires purs sur les fonctions de filtre de preprocess.py.

Ces tests ne dépendent d'aucun fichier de données — ils peuvent tourner sur
n'importe quelle machine, à tout moment, y compris juste après un git clone
avant d'avoir lancé le moindre script d'extraction. Ils valident la LOGIQUE
des fonctions de filtre avec des cas choisis délibérément (dont les cas
limites), pas le résultat réel produit par le pipeline (voir
test_data_quality.py pour ça).

Usage :
    pytest tests/test_preprocess.py -v
"""

from datetime import date, timedelta

import pytest

from src import config
from src.preprocess import _is_in_perimeter, _is_recent_enough

# --------------------------------------------------------------------------
# _is_recent_enough
#
# Les dates de test sont calculées par rapport à aujourd'hui plutôt qu'écrites
# en dur, pour que ces tests restent valides quelle que soit la date à
# laquelle on les exécute (contrairement à test_data_quality.py, qui teste
# volontairement par rapport à la date du jour sur les vraies données — ici
# on veut une logique stable dans le temps, indépendante du calendrier).
# --------------------------------------------------------------------------


def _jours_avant(n: int) -> str:
    return (date.today() - timedelta(days=n)).isoformat()


def _jours_apres(n: int) -> str:
    return (date.today() + timedelta(days=n)).isoformat()


@pytest.mark.parametrize("firstdate_begin, attendu, description", [
    (_jours_avant(10), True, "événement récent (10 jours)"),
    (_jours_avant(364), True, "juste sous la limite (364 jours)"),
    (_jours_avant(config.DAYS_HISTORY), True, "exactement à la limite (>=, doit passer)"),
    (_jours_avant(config.DAYS_HISTORY + 1), False, "juste au-dessus de la limite"),
    (_jours_avant(400), False, "événement ancien (400 jours)"),
    (_jours_apres(30), True, "événement futur (toujours accepté)"),
    (None, False, "champ absent"),
    ("", False, "chaîne vide"),
    ("pas-une-date", False, "date malformée"),
])
def test_is_recent_enough(firstdate_begin, attendu, description):
    assert _is_recent_enough(firstdate_begin) == attendu, description


def test_is_recent_enough_gere_date_avec_heure():
    """firstdate_begin contient normalement une heure (ISO complet, ex:
    '2026-08-01T14:30:00+00:00') — vérifie que seuls les 10 premiers
    caractères (la date) sont utilisés pour la comparaison."""
    date_avec_heure = f"{_jours_avant(5)}T14:30:00+00:00"
    assert _is_recent_enough(date_avec_heure) is True


# --------------------------------------------------------------------------
# _is_in_perimeter
# --------------------------------------------------------------------------

@pytest.mark.parametrize("city, attendu, description", [
    ("Bordeaux", True, "commune du périmètre"),
    ("Mérignac", True, "autre commune du périmètre"),
    ("Villenave-d'Ornon", True, "commune avec apostrophe"),
    ("Libourne", False, "commune hors périmètre (Gironde mais pas Bordeaux Métropole)"),
    ("Paris", False, "ville hors région"),
    (None, False, "champ absent"),
    ("", False, "chaîne vide"),
    ("  Bordeaux  ", True, "espaces superflus (doivent être nettoyés)"),
    ("bordeaux", False, "casse différente (comportement actuel : sensible à la casse)"),
])
def test_is_in_perimeter(city, attendu, description):
    assert _is_in_perimeter(city) == attendu, description


def test_toutes_les_communes_du_perimetre_sont_acceptees():
    """Filet de sécurité : vérifie que chaque commune listée dans la config
    est bien acceptée par la fonction — utile si un jour la liste et la
    fonction de filtre divergent (ex: la fonction change de critère sans
    que la config soit mise à jour en conséquence)."""
    for commune in config.BORDEAUX_METROPOLE_COMMUNES:
        assert _is_in_perimeter(commune) is True, f"{commune} devrait être acceptée"
