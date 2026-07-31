"""
Vérifie que l'environnement nécessaire à l'exécution du chatbot est
disponible : index Faiss construit, clé API Mistral configurée. Sert de
garde-fou rapide en tête de suite de tests — mieux vaut un échec clair ici
qu'une erreur confuse plus loin dans la chaîne (Faiss, appel Mistral...).

Ce fichier remplace test_chatbot_scenarios.py : les scénarios d'interaction
qu'il testait (question pertinente, hors-sujet, vague) sont désormais
couverts par le jeu de questions/réponses annotées (QA_annotees.json) et
son script d'évaluation (evaluate_chatbot.py), qui mesurent la qualité des
réponses plutôt que juste l'absence de plantage.
"""

from src import config


def test_index_faiss_present():
    assert (config.VECTOR_STORE_DIR / "index.faiss").exists(), (
        "Index Faiss introuvable — lance d'abord tout le pipeline "
        "(fetch_raw_data, preprocess, vectorize, build_faiss_index)."
    )


def test_cle_api_mistral_presente():
    assert config.MISTRAL_API_KEY, (
        "MISTRAL_API_KEY absente. Copie .env.example en .env et renseigne ta clé."
    )