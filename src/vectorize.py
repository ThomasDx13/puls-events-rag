"""
Vectorisation des événements propres (data/processed/events.json) :

1. Construit un texte par événement (titre, description(s), conditions,
   mots-clés, lieu, date) — voir build_embedding_text().
2. Découpe ce texte en chunks (RecursiveCharacterTextSplitter, cf. config.py
   pour les valeurs CHUNK_SIZE/CHUNK_OVERLAP et leur justification).
3. Envoie les chunks à l'API Mistral (mistral-embed) par petits batchs, en
   respectant la limite de débit du compte (1 req/s).
4. Sauvegarde le résultat en deux fichiers alignés par position :
   - data/processed/chunks.json    : texte + métadonnées de chaque chunk
   - data/processed/embeddings.npy : matrice numpy des vecteurs correspondants

Ce script NE construit PAS l'index Faiss — ça viendra dans un script séparé
(étape suivante), justement pour ne jamais avoir à repayer des appels Mistral
si on veut retravailler l'index par la suite.

Usage :
    # Test rapide sur un petit échantillon avant de lancer sur la totalité
    python -m src.vectorize --sample 20

    # Vectorisation complète
    python -m src.vectorize
"""

import argparse
import json
import sys
import time
from datetime import datetime

import numpy as np
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

JOURS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MOIS_FR = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]


def format_date_display(event: dict) -> str:
    """Date en français lisible, sans dépendre de la locale du système
    (cf. les soucis Windows déjà rencontrés sur ce projet). Gère le cas d'un
    événement sur plusieurs jours (ex: "du 3 septembre au 15 décembre 2026").
    """
    try:
        debut = datetime.fromisoformat(event["date_start"])
    except (ValueError, TypeError, KeyError):
        return ""

    fin_brut = event.get("date_end") or event["date_start"]
    try:
        fin = datetime.fromisoformat(fin_brut)
    except (ValueError, TypeError):
        fin = debut

    if debut.date() == fin.date():
        return f"{JOURS_FR[debut.weekday()]} {debut.day} {MOIS_FR[debut.month - 1]} {debut.year}"
    return (f"du {debut.day} {MOIS_FR[debut.month - 1]} "
            f"au {fin.day} {MOIS_FR[fin.month - 1]} {fin.year}")


def _ensure_sentence(text: str) -> str:
    """Ajoute un point final si absent, pour uniformiser la ponctuation
    (voir discussion sur pourquoi : aide l'embedding à traiter le texte
    comme de vraies phrases plutôt qu'une concaténation brute de champs)."""
    text = text.rstrip()
    if text and not text.endswith((".", "!", "?")):
        text += "."
    return text


def build_header(event: dict) -> str:
    """Contexte essentiel (quoi/où/quand/mots-clés), préfixé à CHAQUE chunk
    de l'événement — jamais laissé isolé. Voir l'explication du problème des
    chunks orphelins (fragment "titre" ou "lieu/date" tout seul, ~50
    caractères) : en préfixant systématiquement ce header, un chunk retrouvé
    isolément par la recherche reste toujours exploitable, et aucun fragment
    ne se retrouve réduit à quelques mots sans contexte.

    Les mots-clés sont inclus ici (et non dans build_body()) pour concentrer
    leur signal discriminant sur un texte court, présent dans TOUS les
    chunks de l'événement — voir le commentaire détaillé sur la ligne
    correspondante plus bas.
    """
    header = _ensure_sentence(f"{event['title']}. {event['description']}")

    location = event["location"]
    lieu = f"{location['name']}, {location['city']}" if location["name"] else location["city"]
    header += f" Lieu : {lieu}."

    date_display = format_date_display(event)
    if date_display:
        header += f" Date : {date_display}."

    # Mots-clés placés ici (header, préfixé à CHAQUE chunk) plutôt que dans
    # build_body() : deux raisons distinctes, pas juste une préférence de
    # style.
    # 1. Signal concentré : un chunk de ~1500 caractères est majoritairement
    #    composé de texte peu discriminant (adresse, conditions, formules
    #    administratives) — le signal "Metal" y est dilué. Le placer dans le
    #    header (texte court, dense, présent partout) augmente son poids
    #    relatif dans le vecteur d'embedding.
    # 2. Bug structurel évité : dans l'ancienne version, les mots-clés
    #    étaient en fin de body — pour un événement découpé en plusieurs
    #    chunks, seul le DERNIER morceau les contenait. Un chunk retrouvé
    #    isolément parmi les premiers n'avait alors aucune trace des
    #    mots-clés dans son texte embeddé. Le header étant préfixé à chaque
    #    chunk (voir docstring de la fonction), ce problème disparaît.
    if event.get("keywords"):
        header += f" Mots-clés : {', '.join(event['keywords'])}."

    return header


def build_body(event: dict) -> str:
    """Contenu détaillé et potentiellement long — c'est cette partie, et
    UNIQUEMENT celle-ci, qui peut être découpée en plusieurs chunks si elle
    dépasse CHUNK_SIZE. Le header (court, toujours présent) n'entre pas dans
    ce découpage.
    """
    parts = []
    if event.get("long_description"):
        parts.append(_ensure_sentence(event["long_description"]))

    details = []
    if event.get("conditions"):
        details.append(f"Conditions : {_ensure_sentence(event['conditions'])}")
    if details:
        parts.append(" ".join(details))

    return "\n\n".join(parts)


def build_chunk_metadata(event: dict, chunk_index: int, total_chunks: int) -> dict:
    """Métadonnées attachées à chaque chunk — ce sont elles que les tests
    unitaires (étape suivante) valideront (date, ville), et elles servent
    aussi à afficher/filtrer les résultats plus tard dans le chatbot."""
    location = event["location"]
    return {
        "uid": event["uid"],
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "title": event["title"],
        "date_start": event["date_start"],
        "date_end": event["date_end"],
        "city": location["city"],
        "address": location["address"],
        "postal_code": location["postal_code"],
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "url": event["url"],
        "keywords": event["keywords"],
        "conditions": event["conditions"],
        "registration": event["registration"],
        "source_agenda": event["source_agenda"],
    }


def load_events(sample: int | None) -> list[dict]:
    if not config.PROCESSED_DATA_FILE.exists():
        raise FileNotFoundError(
            f"{config.PROCESSED_DATA_FILE} introuvable. "
            "Lance d'abord `python -m src.preprocess`."
        )
    events = json.loads(config.PROCESSED_DATA_FILE.read_text(encoding="utf-8"))
    if sample:
        events = events[:sample]
    return events


def build_chunks(events: list[dict]) -> list[dict]:
    """Découpe chaque événement en chunks. Retourne une liste de dicts
    {text, metadata} — un par chunk, tous événements confondus.

    Le header (titre/lieu/date) est calculé une fois par événement puis
    préfixé à chaque morceau du body — jamais traité comme un chunk à part
    entière. CHUNK_SIZE s'applique donc à la longueur du BODY, pas du texte
    final (header + morceau) : un chunk final peut dépasser légèrement 1500
    caractères une fois le header ajouté (~150-250 caractères de plus selon
    les événements), ce qui reste largement anodin au vu de la fenêtre de
    contexte du modèle (8000 tokens).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for event in events:
        header = build_header(event)
        body = build_body(event)

        body_pieces = splitter.split_text(body) if body else []
        texts = [f"{header}\n\n{piece}" for piece in body_pieces] if body_pieces else [header]

        for i, chunk_text in enumerate(texts):
            all_chunks.append({
                "text": chunk_text,
                "metadata": build_chunk_metadata(event, chunk_index=i, total_chunks=len(texts)),
            })
    return all_chunks


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """Appelle l'API Mistral par batchs, avec pause entre chaque appel pour
    respecter le débit autorisé, et réduction automatique du batch en cas
    d'erreur (taille de batch max non documentée officiellement, cf.
    discussion — on ne suppose pas un chiffre, on s'adapte à ce que l'API
    accepte réellement)."""
    if not config.MISTRAL_API_KEY:
        raise RuntimeError(
            "MISTRAL_API_KEY absente. Copie .env.example en .env et renseigne ta clé "
            "(https://console.mistral.ai)."
        )

    embeddings_client = MistralAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        mistral_api_key=config.MISTRAL_API_KEY,
        max_concurrent_requests=1,  # on gère nous-mêmes le rythme, pas de rafale parallèle
    )

    texts = [c["text"] for c in chunks]
    all_vectors: list[list[float]] = []
    batch_size = config.EMBEDDING_BATCH_SIZE
    i = 0

    while i < len(texts):
        batch = texts[i:i + batch_size]
        for attempt in range(1, config.EMBEDDING_MAX_RETRIES + 1):
            try:
                vectors = embeddings_client.embed_documents(batch)
                all_vectors.extend(vectors)
                i += len(batch)
                print(f"  {i}/{len(texts)} chunks vectorisés")
                break
            except Exception as exc:
                message = str(exc).lower()
                size_related = any(
                    keyword in message for keyword in ("too large", "too many", "size", "limit")
                )
                if size_related and batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    batch = texts[i:i + batch_size]
                    print(f"  [!] Erreur possiblement liée à la taille du batch, "
                          f"réduction à {batch_size} et nouvel essai : {exc}")
                    continue
                if attempt < config.EMBEDDING_MAX_RETRIES:
                    print(f"  [!] Erreur (tentative {attempt}/{config.EMBEDDING_MAX_RETRIES}), "
                          f"nouvel essai dans {2 * attempt}s : {exc}")
                    time.sleep(2 * attempt)
                else:
                    raise RuntimeError(
                        f"Échec définitif au chunk {i} après {config.EMBEDDING_MAX_RETRIES} tentatives : {exc}"
                    )
        time.sleep(config.EMBEDDING_REQUEST_DELAY_SECONDS)

    return np.array(all_vectors, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Ne traiter que les N premiers événements (pour tester avant de lancer sur la totalité).",
    )
    args = parser.parse_args()

    events = load_events(args.sample)
    print(f"{len(events)} événements chargés"
          f"{' (échantillon)' if args.sample else ''} depuis {config.PROCESSED_DATA_FILE}\n")

    chunks = build_chunks(events)
    n_multi = sum(1 for c in chunks if c["metadata"]["total_chunks"] > 1)
    print(f"{len(chunks)} chunks générés "
          f"({n_multi} proviennent d'événements découpés en plusieurs morceaux)\n")

    print("Vectorisation en cours...")
    vectors = embed_chunks(chunks)

    if vectors.shape[0] != len(chunks):
        print(f"\n[!] Attention : {vectors.shape[0]} vecteurs produits pour {len(chunks)} chunks "
              "— désalignement, ne pas sauvegarder tel quel.")
        sys.exit(1)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.CHUNKS_FILE.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(config.EMBEDDINGS_FILE, vectors)

    print(f"\n{len(chunks)} chunks + vecteurs (dimension {vectors.shape[1]}) sauvegardés :")
    print(f"  {config.CHUNKS_FILE}")
    print(f"  {config.EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
