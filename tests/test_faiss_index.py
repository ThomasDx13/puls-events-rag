"""
Vérifie l'index Faiss construit par build_faiss_index.py :
- complétude : tous les événements de events.json sont bien représentés
- pertinence : chercher le titre exact d'un événement le retrouve bien
- vitesse : la recherche Faiss elle-même reste rapide

Comme test_data_quality.py, ce fichier dépend de données déjà générées et
est ignoré (SKIPPED) proprement si l'index n'existe pas encore, plutôt que
de planter.

Contrairement aux autres fichiers de tests du projet, ceux-ci font de VRAIS
appels à l'API Mistral (pour transformer une question de recherche en
vecteur) — volontairement : on veut valider le comportement réel de bout en
bout, pas un comportement simulé. Chaque lancement consomme donc un tout
petit peu de quota API (quelques tokens).

Usage :
    pytest tests/test_faiss_index.py -v
"""

import json
import time

import pytest
from langchain_community.vectorstores import FAISS

from src import config
from src.build_faiss_index import load_vectorstore

# Seuil volontairement large (recherche Faiss typiquement < 5 ms sur ~12 600
# vecteurs avec un index Flat) : ce test vérifie l'absence de dérive grave
# (ex: bascule accidentelle vers un algorithme mal adapté), pas une
# performance de pointe.
SEUIL_VITESSE_MS = 200


def _load_index() -> FAISS:
    if not (config.VECTOR_STORE_DIR / "index.faiss").exists():
        pytest.skip(
            f"Index Faiss introuvable dans {config.VECTOR_STORE_DIR} — lance "
            "d'abord `python -m src.build_faiss_index`."
        )
    if not config.MISTRAL_API_KEY:
        pytest.skip("MISTRAL_API_KEY absente — nécessaire pour interroger l'index.")
    return load_vectorstore()


def _load_chunks() -> list[dict]:
    if not config.CHUNKS_FILE.exists():
        pytest.skip(f"{config.CHUNKS_FILE} introuvable.")
    return json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))


def _load_events() -> list[dict]:
    if not config.PROCESSED_DATA_FILE.exists():
        pytest.skip(f"{config.PROCESSED_DATA_FILE} introuvable.")
    return json.loads(config.PROCESSED_DATA_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def index():
    return _load_index()


@pytest.fixture(scope="module")
def chunks():
    return _load_chunks()


@pytest.fixture(scope="module")
def events():
    return _load_events()


def test_index_contient_autant_de_vecteurs_que_de_chunks(index, chunks):
    assert index.index.ntotal == len(chunks), (
        f"{index.index.ntotal} vecteurs dans l'index pour {len(chunks)} chunks attendus."
    )


def test_tous_les_evenements_sont_representes(index, events):
    """Vérifie qu'aucun événement n'a disparu ENTIÈREMENT de l'index (tous
    ses chunks manquants) — un cas que le test précédent, qui ne compte que
    des chunks au total, ne détecterait pas forcément."""
    uids_events = {e["uid"] for e in events}
    uids_index = {doc.metadata["uid"] for doc in index.docstore._dict.values()}
    manquants = uids_events - uids_index
    assert not manquants, (
        f"{len(manquants)} événement(s) absent(s) de l'index : {list(manquants)[:10]}"
    )


def test_recherche_retrouve_le_bon_evenement(index, chunks):
    """On prend le titre exact d'un vrai chunk indexé, on le cherche, et on
    vérifie qu'il ressort dans le top 5 — valide toute la chaîne (embedding
    de la requête -> recherche -> résultat) avec un vrai appel API."""
    chunk_cible = next(c for c in chunks if c["metadata"]["chunk_index"] == 0)
    titre = chunk_cible["metadata"]["title"]
    uid_attendu = chunk_cible["metadata"]["uid"]

    # similarity_search_with_score (pas _with_relevance_scores) : score brut,
    # fiable pour notre config (MAX_INNER_PRODUCT + vecteurs normalisés =
    # produit scalaire brut = cosinus exact). Voir chatbot.py pour le détail
    # de pourquoi la version "relevance_scores" a été écartée.
    resultats = index.similarity_search_with_score(titre, k=5)
    uids_trouves = [doc.metadata["uid"] for doc, _score in resultats]

    print(f"\n  Titre recherché : {titre}")
    print(f"  Top 5 résultats :")
    for doc, score in resultats:
        print(f"    score={score:.4f}  uid={doc.metadata['uid']:6s}  titre={doc.metadata['title']}")

    assert uid_attendu in uids_trouves, (
        f"L'événement '{titre}' (uid={uid_attendu}) n'apparaît pas dans le "
        f"top 5 pour une recherche sur son propre titre. Résultats obtenus : "
        f"{uids_trouves}"
    )


def test_recherche_faiss_reste_rapide(index):
    """Mesure UNIQUEMENT le temps de recherche Faiss — pas l'appel réseau à
    Mistral pour transformer la question en vecteur (qui domine largement le
    temps total et n'a rien à voir avec la performance de l'index lui-même).
    L'embedding de la requête est donc fait AVANT le chronométrage."""
    vecteur_requete = index.embedding_function.embed_query("concert de musique")

    debut = time.perf_counter()
    index.similarity_search_by_vector(vecteur_requete, k=5)
    duree_ms = (time.perf_counter() - debut) * 1000

    print(f"\n  Recherche Faiss : {duree_ms:.2f} ms sur {index.index.ntotal} vecteurs "
          f"(seuil : {SEUIL_VITESSE_MS} ms)")

    assert duree_ms < SEUIL_VITESSE_MS, (
        f"Recherche Faiss trop lente : {duree_ms:.2f} ms (seuil : {SEUIL_VITESSE_MS} ms) "
        f"sur {index.index.ntotal} vecteurs."
    )


def test_filtrage_par_date_exacte_fonctionne(index, chunks):
    """La recherche SÉMANTIQUE seule ne sait pas comparer des dates
    exactement (un embedding ne fait pas d'arithmétique sur les nombres —
    voir discussion : chercher une date précise par similarité ramène des
    dates "du même genre", pas la date exacte). Le filtrage sur métadonnées,
    lui, fait une comparaison de chaîne stricte — fiable pour un critère
    précis, contrairement à l'embedding. Ce test valide cette approche
    hybride (recherche + filtre structuré), pas la recherche sémantique pure.

    Point important : LangChain n'applique le filtre que sur les `fetch_k`
    voisins sémantiques les plus proches de la requête (20 par défaut), PAS
    sur l'index entier — `k` ne contrôle que le nombre de résultats GARDÉS
    après filtrage, pas le nombre de candidats EXAMINÉS avant. Avec une
    requête neutre, l'événement visé par le filtre pourrait ne pas faire
    partie du top 20 sémantique et le test échouerait à tort. On force donc
    fetch_k au nombre total de chunks pour que le filtre porte réellement
    sur l'intégralité de l'index."""
    chunk_cible = next(c for c in chunks if c["metadata"]["chunk_index"] == 0)
    date_cible = chunk_cible["metadata"]["date_start"][:10]

    # Filtre au niveau du JOUR (pas de l'horodatage exact) : plus réaliste
    # ("les événements du 1er août", pas "l'événement commençant pile à
    # 20h00:00"), et ça permet de vérifier que PLUSIEURS événements
    # partageant une même date sont bien tous retrouvés, pas juste un seul
    # match exact par coïncidence. Le paramètre `filter` de LangChain accepte
    # soit un dict (égalité stricte), soit une fonction — ici une fonction
    # est nécessaire puisqu'on veut comparer un préfixe, pas une égalité.
    resultats = index.similarity_search(
        "evenement",  # requête neutre : c'est le FILTRE qui fait le travail, pas la recherche sémantique
        k=len(chunks),
        fetch_k=len(chunks),
        filter=lambda meta: meta["date_start"][:10] == date_cible,
    )

    print(f"\n  Date recherchée : {date_cible}")
    print(f"  {len(resultats)} événement(s) trouvé(s) pour ce jour :")
    for doc in resultats:
        print(f"    uid={doc.metadata['uid']:6s} titre={doc.metadata['title']}")

    assert len(resultats) > 0, f"Aucun résultat pour le filtre date_start commençant par {date_cible}"
    assert all(doc.metadata["date_start"][:10] == date_cible for doc in resultats), (
        "Le filtre a laissé passer une date différente de celle demandée."
    )


def test_recherche_par_mots_cles_retrouve_des_evenements_pertinents(index, chunks):
    """Cherche avec des mots-clés SEULS (pas le titre) et vérifie qu'au
    moins un résultat du top 5 partage au moins un mot-clé avec la requête.
    Contrairement à test_recherche_retrouve_le_bon_evenement (qui cherche le
    titre EXACT d'un événement précis — donc pas vraiment une recherche
    sémantique, plutôt une quasi-citation), ceci teste une vraie recherche
    thématique où plusieurs événements différents peuvent légitimement
    correspondre."""
    chunk_avec_keywords = next(
        (c for c in chunks if c["metadata"]["chunk_index"] == 0 and c["metadata"].get("keywords")),
        None,
    )
    if chunk_avec_keywords is None:
        pytest.skip("Aucun chunk avec mots-clés trouvé.")

    mots_cles_cible = set(chunk_avec_keywords["metadata"]["keywords"])
    resultats = index.similarity_search_with_score(", ".join(mots_cles_cible), k=5)

    print(f"\n  Mots-clés recherchés : {mots_cles_cible}")
    print(f"  Top 5 résultats :")
    for doc, score in resultats:
        partages = set(doc.metadata.get("keywords") or []) & mots_cles_cible
        print(f"    score={score:.4f}  uid={doc.metadata['uid']:6s}  "
              f"titre={doc.metadata['title']}  mots-clés partagés={partages or '-'}")

    trouve = any(
        set(doc.metadata.get("keywords") or []) & mots_cles_cible
        for doc, _score in resultats
    )
    assert trouve, (
        f"Aucun résultat du top 5 ne partage de mot-clé avec la requête ({mots_cles_cible})."
    )
