"""
Construction de l'index Faiss à partir des chunks et vecteurs déjà calculés
par vectorize.py (data/processed/chunks.json + embeddings.npy).

Contrairement à vectorize.py, cette étape est GRATUITE et LOCALE : aucun
appel à l'API Mistral n'est fait ici (les vecteurs existent déjà sur disque).
Peut être relancée autant de fois que nécessaire (ex: pour essayer un autre
type d'index) sans jamais repayer quoi que ce soit.

Index Flat (recherche exacte, pas d'approximation) : voir le README pour la
justification de ce choix au vu du volume (~12 600 vecteurs).

Stratégie de distance : MAX_INNER_PRODUCT (pas COSINE). Pour des vecteurs
normalisés (norme 1, garanti par Mistral), produit scalaire = similarité
cosinus exacte — mais UNIQUEMENT si on récupère le score BRUT (via
`similarity_search_with_score`), sans passer par
`similarity_search_with_relevance_scores` : sa fonction de conversion pour
MAX_INNER_PRODUCT s'est révélée inversée (un cosinus de 1.0, la meilleure
correspondance possible, donnait un score de 0.0). Voir l'historique de
discussion pour le détail de cette découverte.

Usage :
    python -m src.build_faiss_index
"""

import json

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_mistralai import MistralAIEmbeddings

from src import config


def load_chunks_and_embeddings() -> tuple[list[dict], np.ndarray]:
    if not config.CHUNKS_FILE.exists() or not config.EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"{config.CHUNKS_FILE} et/ou {config.EMBEDDINGS_FILE} introuvable(s). "
            "Lance d'abord `python -m src.vectorize`."
        )
    chunks = json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))
    vectors = np.load(config.EMBEDDINGS_FILE)

    if vectors.shape[0] != len(chunks):
        raise ValueError(
            f"Désalignement : {vectors.shape[0]} vecteurs pour {len(chunks)} chunks. "
            "Les deux fichiers doivent venir de la même exécution de vectorize.py."
        )
    return chunks, vectors


def build_index(chunks: list[dict], vectors: np.ndarray) -> FAISS:
    """Construit l'index à partir des vecteurs déjà calculés (pas de nouvel
    appel API ici). L'objet MistralAIEmbeddings est quand même nécessaire :
    LangChain le conserve pour pouvoir, plus tard, transformer la question
    d'un utilisateur en vecteur au moment de la recherche."""
    if not config.MISTRAL_API_KEY:
        raise RuntimeError(
            "MISTRAL_API_KEY absente. Nécessaire ici pour permettre les futures "
            "recherches (embedding de la question posée par l'utilisateur), même "
            "si aucun appel n'est fait pendant la construction de l'index elle-même."
        )

    embeddings_client = MistralAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        mistral_api_key=config.MISTRAL_API_KEY,
    )

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    text_embeddings = list(zip(texts, vectors.tolist()))

    return FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embeddings_client,
        metadatas=metadatas,
        normalize_L2=True,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def verify_completeness(index: FAISS, chunks: list[dict]) -> None:
    """Vérifie que tous les ÉVÉNEMENTS (pas juste tous les chunks) sont
    représentés dans l'index. Un événement pourrait en théorie disparaître
    entièrement de l'index sans qu'aucune autre vérification ne le remarque
    (ex: un bug ferait sauter tous les chunks d'un même uid)."""
    if not config.PROCESSED_DATA_FILE.exists():
        print("[!] events.json introuvable, vérification de complétude sautée.")
        return

    events = json.loads(config.PROCESSED_DATA_FILE.read_text(encoding="utf-8"))
    uids_events = {e["uid"] for e in events}
    uids_index = {c["metadata"]["uid"] for c in chunks}

    manquants = uids_events - uids_index
    if manquants:
        print(f"[!] {len(manquants)} événement(s) absent(s) de l'index : "
              f"{list(manquants)[:10]}")
    else:
        print(f"Complétude OK : les {len(uids_events)} événements de events.json "
              f"sont tous représentés dans l'index ({index.index.ntotal} vecteurs).")


def load_vectorstore() -> FAISS:
    """Recharge l'index déjà construit et sauvegardé sur disque (contrairement
    à build_index(), qui le construit à partir de zéro). C'est cette fonction
    qu'utilisent les tests et le chatbot — eux n'ont pas besoin de reconstruire
    l'index, juste de le charger tel qu'il a été sauvegardé par main().

    IMPORTANT (bug découvert le 18/07/2026) : FAISS.save_local()/load_local()
    ne persistent PAS distance_strategy ni normalize_L2 — seuls l'index Faiss
    brut et le docstore sont sauvegardés/rechargés. Sans repréciser ces deux
    paramètres explicitement ici, l'index rechargé retombe silencieusement sur
    la stratégie par défaut (EUCLIDEAN_DISTANCE), ce qui ne fausse PAS la
    sélection des résultats (la recherche brute par distance reste équivalente
    au cosinus pour des vecteurs normalisés), mais fausse le SCORE affiché
    pour chaque résultat (mauvaise formule de conversion). D'où : toujours
    repasser les mêmes paramètres qu'à la construction (voir build_index()).
    """
    if not (config.VECTOR_STORE_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            f"Index Faiss introuvable dans {config.VECTOR_STORE_DIR} — lance "
            "d'abord `python -m src.build_faiss_index`."
        )
    if not config.MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY absente — nécessaire pour interroger l'index.")

    embeddings_client = MistralAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        mistral_api_key=config.MISTRAL_API_KEY,
    )
    return FAISS.load_local(
        str(config.VECTOR_STORE_DIR),
        embeddings_client,
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        normalize_L2=True,
    )


def main() -> None:
    chunks, vectors = load_chunks_and_embeddings()
    print(f"{len(chunks)} chunks / vecteurs chargés (dimension {vectors.shape[1]})\n")

    index = build_index(chunks, vectors)
    print(f"Index Faiss construit : {index.index.ntotal} vecteurs\n")

    verify_completeness(index, chunks)

    config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    index.save_local(str(config.VECTOR_STORE_DIR))
    print(f"\nIndex sauvegardé dans {config.VECTOR_STORE_DIR}")


if __name__ == "__main__":
    main()
