"""
Chatbot de recommandation d'événements culturels — pipeline RAG (Retrieval-
Augmented Generation), construit avec LangChain (LCEL).

===========================================================================
CONCEPTS LCEL (LangChain Expression Language) — à lire avant le code
===========================================================================

LangChain appelle "Runnable" tout objet qui sait faire une seule chose :
recevoir une entrée, produire une sortie, via une méthode `.invoke(entree)`.
Un `ChatPromptTemplate`, un modèle de chat (`ChatMistralAI`), une fonction
Python qu'on enveloppe avec `RunnableLambda`... tout ça, ce sont des
Runnables.

L'opérateur `|` (le même symbole que le "pipe" en ligne de commande Unix,
et ce n'est pas un hasard) CHAÎNE deux Runnables : la sortie du premier
devient l'entrée du second, automatiquement.

    chaine = etape_1 | etape_2 | etape_3
    resultat = chaine.invoke(entree_de_depart)

équivaut à :

    resultat = etape_3.invoke(etape_2.invoke(etape_1.invoke(entree_de_depart)))

...mais en beaucoup plus lisible, et ça permet à LangChain de gérer plein de
choses automatiquement en coulisses (streaming, exécution asynchrone,
traces de debug) sans qu'on ait à s'en soucier.

Deux Runnables "utilitaires" qu'on utilise ci-dessous :
- `RunnableLambda(ma_fonction)` : transforme une fonction Python normale en
  Runnable, pour pouvoir l'inclure dans une chaîne avec `|`.
- `RunnableParallel(a=..., b=...)` : exécute PLUSIEURS Runnables sur la
  MÊME entrée, et renvoie un dictionnaire {"a": resultat_a, "b": resultat_b}
  — utile ici pour produire à la fois le texte de contexte (pour le prompt)
  ET la liste des sources (pour les afficher/les renvoyer à l'utilisateur),
  à partir du même résultat de recherche.

===========================================================================
ARCHITECTURE DE CE FICHIER — deux chaînes distinctes
===========================================================================

1. `retrieval_chain` : question (texte) -> {"context": texte formaté pour le
   prompt, "sources": liste de (Document, score, est_hybride)}. Ne touche
   jamais à Mistral pour la génération, uniquement pour transformer la
   question en vecteur (recherche Faiss) — sauf pour la recherche hybride
   complémentaire (période/mot-clé exact), qui ne fait aucun appel API non
   plus (juste un filtre sur les métadonnées déjà en mémoire).

2. `generation_chain` : {"question": ..., "context": ...} -> AIMessage complet
   (pas juste le texte — voir plus bas pourquoi). prompt | modèle.

La fonction `ask()` orchestre les deux : elle appelle `retrieval_chain`,
puis DÉCIDE si ça vaut le coup d'appeler `generation_chain` (garde-fou
anti-hallucination, voir plus bas) — cette décision reste du Python normal
plutôt qu'un maillon de chaîne, parce que "est-ce qu'on continue ou pas" est
un choix, pas une transformation de données.

Usage :
    from src.chatbot import ask
    resultat = ask("Un concert de musique à Bordeaux ce mois-ci ?")
    print(resultat["answer"])
"""

import json
import math
import re
import unicodedata
from datetime import date, timedelta

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_mistralai import ChatMistralAI

from src import config
from src.build_faiss_index import load_vectorstore
from src.vectorize import MOIS_FR

SYSTEM_PROMPT = """Tu es l'assistant de recommandation d'événements culturels de Puls-Events, \
spécialisé sur le périmètre de Bordeaux Métropole.

Règles impératives :
- Réponds UNIQUEMENT à partir des événements listés dans le contexte ci-dessous. \
N'invente jamais un événement qui n'y figure pas, et ne complète jamais un événement \
avec des informations qui ne sont pas dans le contexte.
- Si la question ne porte pas sur la recherche d'un événement (question générale, \
hors sujet), ou si aucun événement du contexte ne répond vraiment à la demande, \
dis-le simplement et clairement. NE PROPOSE JAMAIS un événement du contexte comme \
"suggestion" ou "alternative" sur la seule base d'un mot en commun avec la question \
(par exemple : un événement dont le titre contient "Paris" ne répond PAS à une \
question sur la capitale de la France — ce n'est pas un lien pertinent).
- Pour chaque événement recommandé, mentionne le titre, la date et le lieu. \
Ajoute les conditions d'accès (prix, réservation) si elles sont fournies.
- Nous sommes le {date_du_jour}. Chaque événement du contexte est annoté \
[À VENIR], [EN COURS] ou [TERMINÉ] — c'est cette annotation qui fait foi, ne recalcule \
JAMAIS toi-même le statut d'un événement à partir de ses dates brutes (l'annotation \
tient déjà compte, le cas échéant, d'une date de fin distincte de la date de début — \
un événement qui s'étend sur plusieurs semaines peut être [EN COURS] ou [TERMINÉ] \
même si sa date de DÉBUT est déjà passée). Ne présente JAMAIS un événement annoté \
[TERMINÉ] comme "à venir", "le prochain" ou "encore accessible". Si la question porte \
explicitement sur un événement passé, tu peux bien sûr t'appuyer sur des événements \
[TERMINÉ] du contexte pour répondre.
- Si la question porte sur une période précise ("ce mois-ci", "cette semaine"...) et \
qu'aucun événement du contexte ne correspond exactement à cette période, mais qu'un \
événement du MÊME SUJET existe à une autre date dans le contexte, tu peux le mentionner \
— à condition de le présenter sous un intitulé qui reflète son annotation ([TERMINÉ] -> \
"le dernier en date était...", [À VENIR] -> "le prochain est..."). Ne regroupe JAMAIS un \
événement [TERMINÉ] et un événement [À VENIR] sous un même intitulé ou une même liste \
(par exemple, n'écris jamais "le prochain concert est :" suivi d'une liste qui inclut un \
événement [TERMINÉ] — même annoté "(déjà passé)"). Reste strictement dans les événements \
du contexte (règle précédente) : ne mentionne un événement à une autre date que s'il \
correspond réellement au sujet demandé, jamais sur la seule base d'un mot en commun.
- Réponds en français, de façon naturelle et concise."""

HUMAN_TEMPLATE = """Question : {question}

Événements pertinents trouvés dans la base :
{context}"""


def _dedupe_par_evenement(
    scored_docs: list[tuple[Document, float, bool]],
    max_count: int,
) -> list[tuple[Document, float, bool]]:
    """Un même événement peut avoir plusieurs chunks indexés (~11% des
    événements, cf. étape 2). Le premier chunk rencontré pour un `uid` donné
    dans `scored_docs` est celui gardé — les suivants (même événement, chunk
    différent) sont ignorés, pour ne jamais montrer deux fois le même
    événement au modèle. `max_count` est paramétrable (pas toujours
    RAG_MAX_EVENTS) : on l'utilise aussi pour plafonner séparément les
    résultats hybrides à RAG_MAX_HYBRID_EVENTS, cf. _rechercher_et_dedupliquer.
    """
    vus = set()
    dedupliques = []
    for doc, score, est_hybride in scored_docs:
        uid = doc.metadata["uid"]
        if uid not in vus:
            vus.add(uid)
            dedupliques.append((doc, score, est_hybride))
        if len(dedupliques) >= max_count:
            break
    return dedupliques


def _normaliser(texte: str) -> str:
    """Normalise pour une comparaison insensible à la casse ET aux accents.
    'Metal', 'metal', 'Métal', 'métal' deviennent tous 'metal' après ce
    traitement — la casse ou un accent ne devrait jamais empêcher un
    mot-clé de se reconnaître lui-même (bug réel rencontré : l'événement
    ASHEN avait le mot-clé "Metal" en métadonnées brutes, jamais reconnu
    car comparé sans normalisation à ce stade précis du code — voir
    discussion du 19/07/2026)."""
    texte = unicodedata.normalize("NFKD", texte.lower())
    return "".join(c for c in texte if not unicodedata.combining(c))


def _construire_vocabulaire_mots_cles() -> set[str]:
    """Construit une seule fois (à l'import du module) l'ensemble des
    mots-clés RARES (sous RAG_KEYWORD_MAX_FREQUENCY, cf. config.py),
    à partir de chunks.json. Les mots-clés trop fréquents ("bordeaux",
    "musique", "concert"...) sont exclus volontairement : trouvés le
    19/07/2026, ils se déclenchaient sur presque n'importe quelle question
    et remplissaient les emplacements hybrides avec des événements sans
    rapport avec la vraie demande (ex: "Fête cuivrée" retrouvé pour une
    question sur un concert de METAL, via le mot-clé générique "concert").

    Second filtre, complémentaire : RAG_MOTS_CLES_EXCLUS (config.py) retire
    en plus une liste de mots choisis manuellement, qui restent rares en
    occurrences absolues (donc passent sous le seuil de fréquence) tout en
    étant non-discriminants, car partagés par des catégories d'événements
    sans rapport entre elles (ex: "cours", trouvé le 27/07/2026).

    POC : vocabulaire figé à l'import, pas de rafraîchissement dynamique si
    vectorize.py est relancé sans redémarrer le chatbot — cas jugé trop
    marginal pour la complexité que ça ajouterait (cf. discussion)."""
    if not config.CHUNKS_FILE.exists():
        return set()
    chunks = json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))

    compteur: dict[str, int] = {}
    for chunk in chunks:
        for mot_cle in chunk["metadata"].get("keywords") or []:
            mot_cle_normalise = _normaliser(mot_cle)
            compteur[mot_cle_normalise] = compteur.get(mot_cle_normalise, 0) + 1

    seuil_absolu = len(chunks) * config.RAG_KEYWORD_MAX_FREQUENCY
    vocabulaire = {mot_cle for mot_cle, n in compteur.items() if n <= seuil_absolu}

    # Filtre complémentaire au seuil de fréquence : certains mots restent
    # rares en occurrences absolues tout en étant non-discriminants, parce
    # que partagés par des catégories d'événements sans rapport entre elles
    # (ex: "cours", trouvé le 27/07/2026 sur "Cours de dessin" remonté pour
    # une question sur "bachata"). Voir RAG_MOTS_CLES_EXCLUS dans config.py
    # pour le détail et le principe de sélection.
    return vocabulaire - config.RAG_MOTS_CLES_EXCLUS


_VOCABULAIRE_MOTS_CLES = _construire_vocabulaire_mots_cles()


def _detecter_mots_cles_dans_question(question: str) -> set[str]:
    """Cherche, parmi les mots-clés RÉELLEMENT présents dans la base, ceux
    qui apparaissent dans la question. \\b...\\b (frontière de mot) évite
    qu'un mot-clé court matche par accident à l'intérieur d'un autre mot.
    Question ET vocabulaire sont normalisés (casse + accents, cf.
    _normaliser) avant comparaison."""
    question_normalisee = _normaliser(question)
    return {
        mot_cle for mot_cle in _VOCABULAIRE_MOTS_CLES
        if re.search(rf"\b{re.escape(mot_cle)}\b", question_normalisee)
    }


# --------------------------------------------------------------------------
# Fusion score sémantique + score lexical (TF-IDF), décidée le 27/07/2026.
#
# Objectif : le score sémantique seul se tasse parfois entre des événements
# pertinents et non-pertinents (ex: "Fête cuivrée" à 0,77 contre un vrai
# concert de METAL à 0,74). Un score lexical, calculé mot à mot sur le
# TITRE et les MOTS-CLÉS d'un événement (pas la description complète — même
# raison que pour l'embedding : trop de texte non-discriminant dilue le
# signal), vient compléter ce score plutôt que le remplacer.
#
# Tokenisation MOT PAR MOT ici (via _tokeniser, un set de mots normalisés) —
# différent du reste du fichier : _construire_vocabulaire_mots_cles() et
# _detecter_mots_cles_dans_question() comparent des mots-clés ENTIERS
# ("cours de danse" en un seul bloc), utilisés pour le filtre hybride exact.
# Ici, chaque MOT individuel est pondéré séparément (IDF), pour capter un
# chevauchement partiel de vocabulaire que le filtre hybride, plus strict,
# ne verrait pas du tout.
# --------------------------------------------------------------------------

def _tokeniser(texte: str) -> set[str]:
    """Découpe un texte en un ensemble de mots normalisés (casse + accents,
    cf. _normaliser), sans répétition — pour TF-IDF ici, seule la PRÉSENCE
    d'un mot dans un champ compte (voir _tokens_titre_et_mots_cles), pas son
    nombre brut d'occurrences (nos titres/mots-clés sont trop courts pour
    que la répétition soit un signal fiable, cf. discussion du 27/07/2026).

    Retire aussi les mots vides (RAG_MOTS_VIDES, config.py) : un test sur un
    petit corpus a montré qu'un mot comme "de" n'a pas toujours un IDF assez
    proche de 0 pour être neutralisé tout seul -- il peut fausser le score
    en faveur d'événements qui ne partagent que ce mot de liaison avec la
    question (cf. discussion du 27/07/2026, contredit ce qu'on pensait la
    veille)."""
    return {
        mot for mot in re.findall(r"\w+", _normaliser(texte))
        if mot not in config.RAG_MOTS_VIDES
    }


def _tokens_titre_et_mots_cles(metadata: dict) -> tuple[set[str], set[str]]:
    """Tokens du titre et des mots-clés d'un événement, GARDÉS SÉPARÉS : le
    score TF a besoin de vérifier chaque champ indépendamment (un mot présent
    à la fois dans le titre ET les mots-clés est un signal plus fort qu'un
    mot présent dans un seul des deux, cf. discussion du 27/07/2026)."""
    titre = _tokeniser(metadata.get("title", ""))
    mots_cles = _tokeniser(" ".join(metadata.get("keywords") or []))
    return titre, mots_cles


def _construire_idf_mots() -> dict[str, float]:
    """Construit une seule fois (à l'import, comme _VOCABULAIRE_MOTS_CLES)
    la table IDF (Inverse Document Frequency) de chaque mot apparaissant
    dans un titre ou une liste de mots-clés : IDF(mot) = log(N / df(mot)),
    où N = nombre d'événements UNIQUES et df(mot) = nombre d'événements
    (uniques) où ce mot apparaît. Un mot rare (df faible) obtient un IDF
    élevé -> poids fort dans le score final ; un mot présent partout obtient
    un IDF proche de 0 -> poids quasi nul, sans avoir besoin d'une liste
    d'exclusion manuelle pour ce calcul-là (contrairement à
    RAG_MOTS_CLES_EXCLUS, qui reste nécessaire pour le filtre hybride EXACT,
    un mécanisme différent -- voir sa docstring).

    Compté par ÉVÉNEMENT UNIQUE, pas par chunk (contrairement à
    _construire_vocabulaire_mots_cles(), qui compte par chunk -- ce choix-là
    n'est pas modifié ici, cf. discussion du 27/07/2026) : un même événement
    présent dans plusieurs chunks (~11% des événements, cf. étape 2) ne doit
    compter qu'une fois dans df, sous peine de fausser l'IDF de ses propres
    mots."""
    if not config.CHUNKS_FILE.exists():
        return {}
    chunks = json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))

    evenements_uniques: dict[str, dict] = {}
    for chunk in chunks:
        m = chunk["metadata"]
        evenements_uniques.setdefault(m["uid"], m)

    document_frequency: dict[str, int] = {}
    for m in evenements_uniques.values():
        titre_tokens, mots_cles_tokens = _tokens_titre_et_mots_cles(m)
        for mot in titre_tokens | mots_cles_tokens:
            document_frequency[mot] = document_frequency.get(mot, 0) + 1

    n_evenements = len(evenements_uniques)
    return {mot: math.log(n_evenements / df) for mot, df in document_frequency.items()}


_IDF_MOTS = _construire_idf_mots()


def _score_lexical(question: str, metadata: dict) -> float:
    """Score lexical pondéré IDF, normalisé par le poids total des mots de
    la question (formule et raisonnement complet : discussion du 27/07/2026) :

        score = Σ TF(mot, évènement) × IDF(mot)   [mots communs question/évènement]
                ─────────────────────────────────────────────────────────
                Σ IDF(mot)                         [mots de la question]

    TF(mot, évènement) ∈ {0, 1, 2} : +1 si le mot est dans le titre, +1 s'il
    est dans les mots-clés -- indépendamment l'un de l'autre, PAS un compte
    brut de répétitions (cf. _tokeniser).

    Renvoie 0.0 si aucun mot de la question n'a d'IDF connu (mots absents de
    tout le corpus, ou question vide après tokenisation) -- évite une
    division par zéro plutôt que de laisser planter le chatbot dessus."""
    tokens_question = _tokeniser(question)
    poids_question = sum(_IDF_MOTS.get(mot, 0.0) for mot in tokens_question)
    if poids_question == 0.0:
        return 0.0

    titre_tokens, mots_cles_tokens = _tokens_titre_et_mots_cles(metadata)

    poids_communs = 0.0
    for mot in tokens_question:
        idf = _IDF_MOTS.get(mot)
        if idf is None:
            continue
        tf = (mot in titre_tokens) + (mot in mots_cles_tokens)
        if tf:
            poids_communs += tf * idf

    return poids_communs / poids_question


def _appliquer_bonus_lexical(
    question: str,
    resultats: list[tuple[Document, float, bool]],
) -> list[tuple[Document, float, bool]]:
    """Remplace le score de chaque résultat par
    score_sémantique + RAG_POIDS_BONUS_LEXICAL × score_lexical (décision du
    27/07/2026 : un seul score, cohérent partout dans le pipeline -- garde-
    fou anti-hallucination, tri, affichage -- plutôt que deux scores
    distincts, un pour trier en interne et un pour l'affichage).

    Implication à surveiller : RAG_RELEVANCE_THRESHOLD (config.py) a été
    calibré sur le score sémantique BRUT. Le bonus le fait mécaniquement
    remonter -- un réajustement empirique de ce seuil, après quelques tests,
    est probablement nécessaire (cf. discussion)."""
    return [
        (
            doc,
            score + config.RAG_POIDS_BONUS_LEXICAL * _score_lexical(question, doc.metadata),
            est_hybride,
        )
        for doc, score, est_hybride in resultats
    ]


def _resoudre_annee_probable(mois: int, jour: int) -> int:
    """Une question du style '16 novembre' ne précise pas l'année. On
    suppose la prochaine occurrence à venir : si le 16 novembre de cette
    année est déjà passé, on suppose l'année suivante."""
    aujourdhui = date.today()
    try:
        candidate = date(aujourdhui.year, mois, jour)
    except ValueError:
        return aujourdhui.year
    return aujourdhui.year if candidate >= aujourdhui else aujourdhui.year + 1


def _detecter_date_dans_question(question: str) -> str | None:
    """Détecte un motif 'JJ mois [AAAA]' en français (ex: '16 novembre',
    '16 novembre 2026') et le convertit en date ISO (AAAA-MM-JJ).

    Reste volontairement limitée à ce format explicite (une seule date
    ponctuelle) — les plages et expressions relatives ('ce week-end',
    'entre le X et le Y', 'dans le mois qui vient'...) sont gérées à part,
    voir _detecter_periode_dans_question(), qui appelle cette fonction en
    dernier recours. Split conservé : cette fonction reste utilisable seule
    partout où une date ponctuelle exacte suffit (voir aussi
    test_faiss_index.py)."""
    question_lower = question.lower()
    for mois_index, mois_nom in enumerate(MOIS_FR, start=1):
        match = re.search(rf"\b(\d{{1,2}})\s+{mois_nom}(?:\s+(\d{{4}}))?\b", question_lower)
        if match:
            jour = int(match.group(1))
            annee = int(match.group(2)) if match.group(2) else _resoudre_annee_probable(mois_index, jour)
            try:
                return date(annee, mois_index, jour).isoformat()
            except ValueError:
                return None
    return None


# --------------------------------------------------------------------------
# Détection de PÉRIODE (par opposition à une date ponctuelle unique, cf.
# _detecter_date_dans_question ci-dessus) — ajoutée le 30/07/2026 suite à
# l'évaluation sur QA_annotees.json : les questions à mention temporelle
# relative ou en plage ("aujourd'hui", "ce mois-ci", "entre le 1er et le 15
# octobre"...) ne remontaient JAMAIS le bon contexte, le retrieval n'ayant
# aucune notion de calendrier en dehors d'une date exacte littérale. Même
# philosophie que le reste du fichier : du Python déterministe, aucun appel
# API, un filtre exact sur les métadonnées déjà en mémoire.
# --------------------------------------------------------------------------

MOIS_PATTERN = "|".join(MOIS_FR)


def _detecter_plage_explicite(question: str) -> tuple[str, str] | None:
    """Détecte 'entre le JJ [mois] et le JJ mois [AAAA]'. Le mois de la
    première date est optionnel dans la regex : à l'oral/écrit, on élide
    presque toujours celui de la première date quand les deux dates sont
    dans le même mois ('entre le 1er et le 15 octobre', jamais 'entre le
    1er octobre et le 15 octobre') — si absent, on emprunte celui de la
    seconde date."""
    q = question.lower()
    match = re.search(
        rf"entre\s+le\s+(\d{{1,2}})(?:er)?\s*(?:({MOIS_PATTERN}))?\s*"
        rf"et\s+le\s+(\d{{1,2}})(?:er)?\s+({MOIS_PATTERN})(?:\s+(\d{{4}}))?",
        q,
    )
    if not match:
        return None

    jour1, mois1_nom, jour2, mois2_nom, annee_str = match.groups()
    jour1, jour2 = int(jour1), int(jour2)
    mois1_nom = mois1_nom or mois2_nom
    mois1_index = MOIS_FR.index(mois1_nom) + 1
    mois2_index = MOIS_FR.index(mois2_nom) + 1

    if annee_str:
        annee1 = annee2 = int(annee_str)
    else:
        annee2 = _resoudre_annee_probable(mois2_index, jour2)
        annee1 = annee2
        if date(annee1, mois1_index, jour1) > date(annee2, mois2_index, jour2):
            annee1 -= 1  # plage à cheval sur le nouvel an (ex: 20 déc. -> 5 jan.)

    try:
        return date(annee1, mois1_index, jour1).isoformat(), date(annee2, mois2_index, jour2).isoformat()
    except ValueError:
        return None


def _plage_semaine_calendaire(aujourdhui: date) -> tuple[date, date]:
    """'cette semaine' : aujourd'hui -> dimanche de la semaine calendaire
    en cours (semaine lundi-dimanche)."""
    jours_jusqua_dimanche = (6 - aujourdhui.weekday()) % 7
    return aujourdhui, aujourdhui + timedelta(days=jours_jusqua_dimanche)


def _plage_weekend_a_venir(aujourdhui: date) -> tuple[date, date]:
    """'ce week-end'/'le week-end prochain' : samedi-dimanche à venir. Si on
    est déjà samedi, le week-end en cours ; si on est dimanche, le SUIVANT
    (celui en cours se termine aujourd'hui même, plus la peine de le
    chercher)."""
    jours_jusqua_samedi = (5 - aujourdhui.weekday()) % 7
    samedi = aujourdhui + timedelta(days=jours_jusqua_samedi)
    return samedi, samedi + timedelta(days=1)


def _fin_mois_calendaire(aujourdhui: date) -> date:
    """Dernier jour du mois calendaire en cours (pour 'ce mois-ci')."""
    premier_jour_mois_suivant = date(
        aujourdhui.year + (aujourdhui.month == 12),
        (aujourdhui.month % 12) + 1, 1,
    )
    return premier_jour_mois_suivant - timedelta(days=1)


# Expressions relatives reconnues -> fonction qui calcule (date_min, date_max)
# à partir d'aujourd'hui. Comparaison faite sur la question normalisée
# (_normaliser, casse + accents), comme le reste du fichier -- pas besoin de
# variantes accentuées ici. Sémantiques décidées le 30/07/2026 :
# - "dans le mois"/"dans les semaines" et "dans la semaine"/"dans les jours"
#   sont volontairement DIFFÉRENTS : les premiers visent une fenêtre large
#   et approximative (30 jours), les seconds une fenêtre courte et précise
#   (6 jours) -- distinction fine mais réelle à l'usage en français.
# - "cette semaine" (semaine calendaire, jusqu'à dimanche) est également
#   DIFFÉRENT de "dans la semaine" (6 jours glissants) : la première ancre
#   sur le calendrier, la seconde sur la date du jour.
EXPRESSIONS_RELATIVES = {
    "aujourd'hui":          lambda ajd: (ajd, ajd),
    "dans la journee":      lambda ajd: (ajd, ajd),
    "ce mois-ci":           lambda ajd: (ajd, _fin_mois_calendaire(ajd)),
    "dans le mois":         lambda ajd: (ajd, ajd + timedelta(days=30)),
    "dans les semaines":    lambda ajd: (ajd, ajd + timedelta(days=30)),
    "cette semaine":        lambda ajd: _plage_semaine_calendaire(ajd),
    "dans la semaine":      lambda ajd: (ajd, ajd + timedelta(days=6)),
    "dans les jours":       lambda ajd: (ajd, ajd + timedelta(days=6)),
    "ce weekend":           lambda ajd: _plage_weekend_a_venir(ajd),
    "ce week-end":          lambda ajd: _plage_weekend_a_venir(ajd),
    "le weekend prochain":  lambda ajd: _plage_weekend_a_venir(ajd),
    "le week-end prochain": lambda ajd: _plage_weekend_a_venir(ajd),
}


def _detecter_periode_dans_question(question: str) -> tuple[str, str] | None:
    """Détecte une période dans la question, renvoyée en (date_min, date_max)
    ISO — trois niveaux, du plus spécifique au plus générique :
      1. Plage explicite ('entre le 1er et le 15 octobre')
      2. Expression relative connue (EXPRESSIONS_RELATIVES)
      3. Date unique ('16 novembre', _detecter_date_dans_question) --
         encapsulée dans le même format (date, date), pour que
         _recherche_hybride_complementaire n'ait qu'UN SEUL filtre de
         date à appliquer, quel que soit le cas de figure.
    Renvoie None si rien n'est détecté (comportement identique à avant :
    aucun filtre de date appliqué à la recherche hybride)."""
    plage_explicite = _detecter_plage_explicite(question)
    if plage_explicite:
        return plage_explicite

    question_normalisee = _normaliser(question)
    for expression, calcule_plage in EXPRESSIONS_RELATIVES.items():
        if re.search(rf"\b{re.escape(expression)}\b", question_normalisee):
            debut, fin = calcule_plage(date.today())
            return debut.isoformat(), fin.isoformat()

    date_unique = _detecter_date_dans_question(question)
    return (date_unique, date_unique) if date_unique else None


def _chevauche_periode(metadata: dict, date_min: str, date_max: str) -> bool:
    """Un événement est concerné par la période demandée si son intervalle
    [date_start, date_end] CHEVAUCHE [date_min, date_max] -- pas seulement
    si date_start tombe dedans. Sans cette distinction, un événement déjà en
    cours (commencé avant la période mais toujours d'actualité pendant
    celle-ci, ex: une exposition qui dure plusieurs mois) ne remonterait
    jamais (cf. discussion du 30/07/2026)."""
    debut = (metadata.get("date_start") or "")[:10]
    if not debut:
        return False
    fin = (metadata.get("date_end") or debut)[:10]  # ponctuel si pas de date_end
    return debut <= date_max and fin >= date_min


def _recherche_hybride_complementaire(question: str, vectorstore) -> list[tuple[Document, float, bool]]:
    """Recherche sémantique classique (Faiss) : fiable pour le sens général,
    mais incapable de garantir qu'un critère PRÉCIS (date exacte, mot-clé
    exact) soit bien représenté dans le lot de candidats — un événement
    pertinent peut être classé très loin dans le tri sémantique et ne
    jamais atteindre le contexte montré à Mistral (cas vérifié : un concert
    de metal classé 3106e sur une question qui le concernait directement).

    Cette fonction complète la recherche sémantique par du FILTRAGE EXACT
    sur les métadonnées (comme pour les dates dans test_faiss_index.py,
    généralisé ici aux mots-clés et intégré au chatbot lui-même — jusqu'ici
    ce filtre n'existait QUE dans les tests, jamais utilisé par le chatbot
    en conditions réelles). `fetch_k` égal au nombre total de vecteurs :
    le filtre doit porter sur l'index ENTIER, pas seulement sur les voisins
    sémantiques les plus proches (piège déjà rencontré à l'étape 3).

    Le filtre de date couvre en réalité une PÉRIODE, pas seulement une date
    exacte (voir _detecter_periode_dans_question) : une date ponctuelle
    ('16 novembre') reste un cas particulier de plage (date, date), mais
    'aujourd'hui', 'ce week-end', 'entre le 1er et le 15 octobre' sont
    maintenant couverts par le même chemin.

    Chaque résultat est tagué `est_hybride=True` — utilisé plus loin pour
    lui garantir une place indépendamment de son score sémantique (souvent
    mauvais par construction, cf. discussion) et pour ne pas le pénaliser
    dans le garde-fou anti-hallucination de ask().
    """
    resultats: list[tuple[Document, float, bool]] = []
    n = vectorstore.index.ntotal

    mots_cles = _detecter_mots_cles_dans_question(question)
    if mots_cles:
        bruts = vectorstore.similarity_search_with_score(
            question, k=n, fetch_k=n,
            filter=lambda meta: bool(
                {_normaliser(k) for k in (meta.get("keywords") or [])} & mots_cles
            ),
        )
        resultats += [(doc, score, True) for doc, score in bruts]

    periode = _detecter_periode_dans_question(question)
    if periode:
        date_min, date_max = periode
        bruts = vectorstore.similarity_search_with_score(
            question, k=n, fetch_k=n,
            filter=lambda meta: _chevauche_periode(meta, date_min, date_max),
        )
        resultats += [(doc, score, True) for doc, score in bruts]

    return resultats


def _rechercher_et_dedupliquer(question: str) -> list[tuple[Document, float, bool]]:
    """Premier maillon de retrieval_chain : recherche sémantique dans Faiss
    (avec scores, indispensables pour le garde-fou anti-hallucination),
    complétée par la recherche hybride (date/mots-clés exacts, voir
    _recherche_hybride_complementaire), puis dédoublonnage. RAG_FETCH_K (100)
    est volontairement plus grand que RAG_MAX_EVENTS (10) : ça laisse de la
    marge pour que le dédoublonnage puisse quand même remonter 10 événements
    UNIQUES, même si plusieurs chunks du même événement occupaient des
    places dans le top 100 brut.

    similarity_search_with_score() (pas similarity_search_with_relevance_
    scores()) : ce dernier applique une fonction de conversion censée
    normaliser le score entre 0 et 1, mais qui s'est révélée incorrecte pour
    notre configuration (MAX_INNER_PRODUCT) — un score brut de 1.0 (identité
    parfaite) donnait un "score de pertinence" de 0.0, littéralement inversé.
    Le score BRUT, lui, est fiable : pour des vecteurs normalisés (norme 1,
    garanti par Mistral), le produit scalaire brut renvoyé par Faiss EST
    exactement la similarité cosinus, sans transformation nécessaire.

    Depuis le 27/07/2026, le score renvoyé dans chaque tuple n'est PLUS une
    similarité cosinus pure : _appliquer_bonus_lexical() y ajoute un bonus
    lexical (TF-IDF sur titre+mots-clés, pondéré par RAG_POIDS_BONUS_LEXICAL)
    avant que cette fonction ne retourne son résultat. Décision assumée : un
    seul score cohérent dans tout le pipeline plutôt que deux scores
    distincts (voir discussion).
    """
    vectorstore = load_vectorstore()

    resultats_hybrides = _recherche_hybride_complementaire(question, vectorstore)
    resultats_semantiques_bruts = vectorstore.similarity_search_with_score(
        question, k=config.RAG_FETCH_K
    )
    resultats_semantiques = [(doc, score, False) for doc, score in resultats_semantiques_bruts]

    # Fusion du score lexical (TF-IDF) dans les DEUX branches, décidée le
    # 27/07/2026 -- voir _appliquer_bonus_lexical. Puis retri obligatoire :
    # le bonus n'est PAS uniforme d'un candidat à l'autre (il dépend de
    # l'overlap de vocabulaire propre à chaque événement), donc l'ordre
    # d'origine (score sémantique seul, renvoyé par Faiss) n'est plus
    # forcément le bon une fois le bonus ajouté. Sans ce retri, le bonus ne
    # changerait que les chiffres affichés -- _dedupe_par_evenement
    # tronquerait toujours sur l'ancien ordre, et rien ne changerait
    # réellement à la sélection finale.
    resultats_hybrides = _appliquer_bonus_lexical(question, resultats_hybrides)
    resultats_semantiques = _appliquer_bonus_lexical(question, resultats_semantiques)
    resultats_hybrides.sort(key=lambda item: item[1], reverse=True)
    resultats_semantiques.sort(key=lambda item: item[1], reverse=True)

    # Sous-plafond AVANT fusion : sans ça, un mot-clé fréquent (ex: "musique")
    # pourrait faire remonter plus de RAG_MAX_EVENTS résultats hybrides à lui
    # seul, monopolisant tous les emplacements finaux et empêchant la
    # recherche sémantique d'apporter quoi que ce soit — même si elle avait
    # de meilleurs candidats à proposer. RAG_MAX_HYBRID_EVENTS (5) garantit
    # qu'au moins RAG_MAX_EVENTS - RAG_MAX_HYBRID_EVENTS (5) emplacements
    # restent réservés à la recherche sémantique classique.
    hybrides_plafonnes = _dedupe_par_evenement(resultats_hybrides, config.RAG_MAX_HYBRID_EVENTS)

    # Les résultats hybrides sont placés EN PREMIER, volontairement, PAS
    # triés ensemble avec les résultats sémantiques par score. Un événement
    # retrouvé par mot-clé/date exact peut avoir un très mauvais score
    # sémantique (c'est justement pour rattraper ce cas qu'on le cherche
    # autrement) — le trier avec les résultats sémantiques le renverrait
    # tout en bas de la liste et l'exclurait du top RAG_MAX_EVENTS, annulant
    # l'intérêt de la recherche hybride. En le plaçant en tête, on garantit
    # sa présence dans le contexte final, indépendamment de son score.
    fusion = hybrides_plafonnes + resultats_semantiques
    return _dedupe_par_evenement(fusion, config.RAG_MAX_EVENTS)


def _statut_temporel(metadata: dict) -> str | None:
    """Calcule en Python si un événement est [À VENIR], [EN COURS] ou
    [TERMINÉ] par rapport à aujourd'hui — plutôt que de laisser Mistral
    comparer lui-même une date de fin qu'il doit d'abord repérer dans du
    texte libre. Ajoutée le 30/07/2026 suite à un cas réel observé sur
    l'évaluation (QA_annotees.json, question sur des cours de danse) : des
    cours terminés depuis juin étaient présentés comme "encore accessibles"
    -- seule date_start (largement dans le passé, mais pas la date qui
    compte pour juger si un événement À PLAGE est encore valide) était
    exposée dans le contexte structuré, la date de fin réelle n'existant
    que noyée dans le texte libre de la description.

    Renvoie None si date_start est absente (ne devrait pas arriver en
    pratique, mais évite de planter dessus)."""
    debut = (metadata.get("date_start") or "")[:10]
    if not debut:
        return None
    fin = (metadata.get("date_end") or debut)[:10]

    aujourdhui = date.today().isoformat()
    if fin < aujourdhui:
        return "TERMINÉ"
    if debut > aujourdhui:
        return "À VENIR"
    return "EN COURS"


def _formater_contexte(scored_docs: list[tuple[Document, float, bool]]) -> str:
    """Transforme la liste d'événements trouvés en texte lisible, inséré
    dans le prompt envoyé à Mistral. Un événement par paragraphe numéroté.

    Chaque événement est annoté [À VENIR]/[EN COURS]/[TERMINÉ] (calculé par
    _statut_temporel, PAS laissé à la charge du modèle -- voir SYSTEM_PROMPT,
    qui instruit Mistral à se fier à cette annotation plutôt qu'à recalculer
    lui-même à partir des dates brutes). La date de fin est également
    affichée explicitement quand elle diffère de la date de début -- avant
    ce correctif, elle n'existait que dans le texte libre de la description,
    jamais dans le bloc structuré (cf. discussion du 30/07/2026)."""
    if not scored_docs:
        return "(aucun événement trouvé)"

    blocs = []
    for i, (doc, _score, _est_hybride) in enumerate(scored_docs, start=1):
        m = doc.metadata
        statut = _statut_temporel(m)
        titre_ligne = f"{i}. {m.get('title', '?')}"
        if statut:
            titre_ligne += f" [{statut}]"

        date_debut = (m.get("date_start") or "?")[:10]
        date_fin = (m.get("date_end") or "")[:10]

        bloc = f"{titre_ligne}\n   Date : {date_debut}"
        if date_fin and date_fin != date_debut:
            bloc += f" (jusqu'au {date_fin})"
        bloc += f"\n   Lieu : {m.get('city', '?')}"

        if m.get("conditions"):
            bloc += f"\n   Conditions : {m['conditions']}"
        bloc += f"\n   Détails : {doc.page_content}"
        blocs.append(bloc)
    return "\n\n".join(blocs)


# --------------------------------------------------------------------------
# Chaîne 1 : récupération (Faiss). Prend une question (str) en entrée,
# renvoie {"context": str, "sources": list[(Document, float)]}.
#
# RunnableLambda(_rechercher_et_dedupliquer) : la recherche + dédoublonnage,
# comme expliqué plus haut.
# RunnableParallel(...) : à partir de CE résultat, produit DEUX choses en
# parallèle : le texte formaté (pour le prompt) et la liste brute (pour les
# sources qu'on affichera/renverra à l'utilisateur) — sans refaire la
# recherche deux fois.
# --------------------------------------------------------------------------
retrieval_chain = RunnableLambda(_rechercher_et_dedupliquer) | RunnableParallel(
    context=RunnableLambda(_formater_contexte),
    sources=RunnableLambda(lambda scored_docs: scored_docs),
)


# --------------------------------------------------------------------------
# Chaîne 2 : génération (Mistral). Prend {"question": str, "context": str}
# en entrée, renvoie un AIMessage (pas juste du texte, volontairement — voir
# plus bas : on a besoin de `.usage_metadata` pour suivre la consommation
# de tokens, une info qu'un StrOutputParser() aurait jetée en ne gardant que
# le texte de la réponse).
#
# prompt : assemble le système + la question + le contexte en une liste de
#          messages (format attendu par un modèle de chat).
# llm    : envoie ces messages à mistral-medium-latest, reçoit la réponse.
# --------------------------------------------------------------------------
_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_TEMPLATE),
])

if not config.MISTRAL_API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY absente. Nécessaire pour la génération des réponses "
        "(en plus de la recherche). Copie .env.example en .env et renseigne ta clé."
    )

_llm = ChatMistralAI(
    model=config.RAG_CHAT_MODEL,
    mistral_api_key=config.MISTRAL_API_KEY,
    temperature=config.RAG_TEMPERATURE,
)
generation_chain = _prompt | _llm


def ask(question: str) -> dict:
    """Point d'entrée du chatbot. Pas d'historique de conversation
    (conforme à la consigne du POC) : chaque appel est indépendant, aucun
    état n'est conservé entre deux questions.

    Retourne {"answer": str, "sources": list[(Document, float, bool)],
    "usage": dict|None} — le 3e élément de chaque source indique si elle
    vient de la recherche hybride (date/mot-clé exact) ou de la recherche
    sémantique classique. "usage" contient input_tokens/output_tokens/
    total_tokens (utile pour surveiller la consommation, vu la limite
    serrée du compte : 25 000 tokens/minute), et vaut None si Mistral n'a
    pas été appelé (cas du garde-fou anti-hallucination, aucun coût).

    Distinction passé/à venir : le retrieval ne filtre JAMAIS par date (un
    événement passé reste une réponse légitime à une question qui porte
    explicitement sur le passé, ex: "quel groupe jouait au bar X la semaine
    dernière ?"). C'est Mistral qui décide, à la génération, si un événement
    du contexte doit être présenté comme à venir ou déjà passé — rendu
    possible en lui donnant la date du jour via {date_du_jour} dans
    SYSTEM_PROMPT (voir discussion du 23/07/2026 : un filtre côté retrieval
    aurait cassé ce cas d'usage légitime sur le passé).
    """
    retrieval_result = retrieval_chain.invoke(question)
    sources = retrieval_result["sources"]

    # Le garde-fou anti-hallucination ne s'applique QUE s'il n'y a AUCUN
    # résultat hybride. Un résultat hybride (date/mot-clé exact) est une
    # preuve de pertinence en soi, indépendante de son score sémantique —
    # qui est souvent mauvais par construction (c'est précisément pour
    # rattraper ce cas qu'on fait cette recherche complémentaire). Sans
    # cette distinction, un résultat hybride bien réel mais mal noté
    # sémantiquement ferait baisser max(tous les scores) et bloquerait
    # Mistral à tort.
    a_un_match_hybride = any(est_hybride for _, _, est_hybride in sources)
    meilleur_score_semantique = max(
        (score for _, score, est_hybride in sources if not est_hybride),
        default=0.0,
    )

    if not a_un_match_hybride and meilleur_score_semantique < config.RAG_RELEVANCE_THRESHOLD:
        # Garde-fou anti-hallucination : rien d'assez pertinent trouvé, on
        # ne consulte MÊME PAS Mistral plutôt que de risquer une réponse
        # improvisée à partir d'un contexte hors-sujet.
        return {"answer": config.RAG_NO_RESULTS_MESSAGE, "sources": [], "usage": None}

    ai_message = generation_chain.invoke({
        "question": question,
        "context": retrieval_result["context"],
        # Recalculé à CHAQUE appel de ask() (pas au chargement du module) :
        # le process Streamlit/CLI peut tourner plusieurs jours, la date du
        # jour à l'import serait vite obsolète. LangChain résout les
        # placeholders {date_du_jour} du prompt (système ET humain) à partir
        # de ce dict, au moment de .invoke() — voir SYSTEM_PROMPT plus haut.
        "date_du_jour": date.today().isoformat(),
    })
    return {
        "answer": ai_message.content,
        "sources": sources,
        "usage": ai_message.usage_metadata,
    }
