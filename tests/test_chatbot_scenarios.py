"""
Scénarios d'interaction avec le chatbot, pour vérifier sa robustesse (point
de vigilance de la consigne : "tester plusieurs scénarios d'interaction").

Contrairement à test_data_quality.py/test_faiss_index.py, la justesse d'une
réponse en langage naturel ne peut pas se vérifier avec un simple `assert`
— il n'existe pas de jeu de questions/réponses annoté pour comparer contre
une vérité de référence (ce sera l'objet de l'étape 5). Ce fichier vérifie
donc ce qui EST vérifiable automatiquement (le chatbot ne plante pas, le
garde-fou anti-hallucination se déclenche bien sur une question hors-sujet),
et affiche le reste pour une relecture humaine (avec `-v -s`).

Usage :
    pytest tests/test_chatbot_scenarios.py -v -s
"""

import pytest

from src import config
from src.chatbot import ask


def _verifier_pipeline_disponible():
    if not (config.VECTOR_STORE_DIR / "index.faiss").exists():
        pytest.skip(
            f"Index Faiss introuvable — lance d'abord tout le pipeline "
            f"(fetch_raw_data, preprocess, vectorize, build_faiss_index)."
        )
    if not config.MISTRAL_API_KEY:
        pytest.skip("MISTRAL_API_KEY absente.")


def test_question_pertinente_donne_une_reponse_avec_sources():
    """Scénario 1 : une question claire, dans le périmètre du projet.
    On vérifie ce qui est objectivement vérifiable (réponse non vide,
    sources trouvées, format correct) — pas la qualité rédactionnelle de la
    réponse elle-même, à relire à l'œil avec -s."""
    _verifier_pipeline_disponible()

    resultat = ask("Je cherche un concert de musique à Bordeaux.")

    print(f"\n\n  Question : Je cherche un concert de musique à Bordeaux.")
    print(f"  Réponse  : {resultat['answer']}")
    print(f"  Sources  :")
    for doc, score, est_hybride in resultat["sources"]:
        marqueur = " [hybride]" if est_hybride else ""
        print(f"    - {doc.metadata['title']} (score={score:.2f}{marqueur})")
    if resultat["usage"]:
        print(f"  Tokens   : {resultat['usage']}")

    assert resultat["answer"], "La réponse ne devrait pas être vide."
    assert resultat["sources"], (
        "Une question sur la musique à Bordeaux devrait trouver au moins "
        "un événement, vu le volume de la base (12 600+ chunks)."
    )


def test_question_hors_sujet_ne_declenche_pas_le_llm():
    """Scénario 2 : une question totalement hors périmètre (rien à voir
    avec des événements culturels à Bordeaux). Le garde-fou anti-
    hallucination (voir chatbot.py) doit renvoyer le message standard SANS
    appeler Mistral pour générer une réponse improvisée."""
    _verifier_pipeline_disponible()

    resultat = ask("Quelle est la capitale du Japon ?")

    print(f"\n\n  Question : Quelle est la capitale du Japon ?")
    print(f"  Réponse  : {resultat['answer']}")
    if resultat["sources"]:
        print(f"  Scores trouvés (ne devrait normalement pas arriver ici) :")
        for doc, score, est_hybride in resultat["sources"]:
            marqueur = " [hybride]" if est_hybride else ""
            print(f"    - {doc.metadata['title']} : {score:.2f}{marqueur}")
    else:
        print(f"  Aucune source (garde-fou déclenché, comme attendu).")

    assert resultat["answer"] == config.RAG_NO_RESULTS_MESSAGE, (
        "Une question totalement hors-sujet devrait déclencher le message "
        "standard, pas une réponse générée. Si ce test échoue, le seuil "
        f"RAG_RELEVANCE_THRESHOLD ({config.RAG_RELEVANCE_THRESHOLD}) est "
        "probablement à ajuster — regarde le score affiché ci-dessus."
    )
    assert resultat["sources"] == []


def test_question_vague():
    """Scénario 3 : une question vague/ambiguë, sans réponse "correcte"
    évidente. Pas d'assertion stricte ici (aucune règle simple ne définirait
    une bonne réponse) — juste un affichage pour relecture humaine, à
    évaluer qualitativement une fois vu."""
    _verifier_pipeline_disponible()

    resultat = ask("Qu'est-ce qu'il y a à faire ce week-end ?")

    print(f"\n\n  Question : Qu'est-ce qu'il y a à faire ce week-end ?")
    print(f"  Réponse  : {resultat['answer']}")
    print(f"  Nombre de sources : {len(resultat['sources'])}")
    print(f"  (Scénario sans assertion stricte — à relire à l'œil)")

    assert resultat["answer"], "La réponse ne devrait pas être vide, même pour une question vague."
