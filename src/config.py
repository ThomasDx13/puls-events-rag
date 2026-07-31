"""
Configuration centralisée du POC Puls-Events RAG.

Modifier ce fichier suffit pour changer le périmètre géographique,
la fenêtre temporelle, ou les chemins de sortie — aucun autre script
ne doit contenir de valeur en dur.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Source de données (Opendatasoft — miroir public du dataset OpenAgenda)
# --------------------------------------------------------------------------
ODS_DOMAIN = "public.opendatasoft.com"
DATASET_ID = "evenements-publics-openagenda"
API_BASE_URL = f"https://{ODS_DOMAIN}/api/explore/v2.1/catalog/datasets/{DATASET_ID}/records"

# --------------------------------------------------------------------------
# Noms de champs côté API.
#
# Confirmés le 16/07/2026 par un appel réel à l'API (Thomas) :
# uid, title_fr, description_fr, longdescription_fr, keywords_fr,
# firstdate_begin, firstdate_end, lastdate_begin, lastdate_end,
# location_name, location_address, location_city, location_postalcode,
# location_coordinates (= {"lat": ..., "lon": ...}), originagenda_title,
# canonicalurl (et NON "link", corrigé après vérification).
# --------------------------------------------------------------------------
FIELD_UID = "uid"
FIELD_TITLE = "title_fr"
FIELD_DESCRIPTION = "description_fr"
FIELD_LONGDESCRIPTION = "longdescription_fr"
FIELD_KEYWORDS = "keywords_fr"
FIELD_FIRSTDATE_BEGIN = "firstdate_begin"
FIELD_FIRSTDATE_END = "firstdate_end"
FIELD_LASTDATE_BEGIN = "lastdate_begin"
FIELD_LASTDATE_END = "lastdate_end"
FIELD_TIMINGS = "timings"
FIELD_LOCATION_NAME = "location_name"
FIELD_LOCATION_ADDRESS = "location_address"
FIELD_LOCATION_CITY = "location_city"
FIELD_LOCATION_POSTALCODE = "location_postalcode"
FIELD_LOCATION_COORDINATES = "location_coordinates"
FIELD_CANONICAL_URL = "canonicalurl"
FIELD_ORIGINAGENDA_TITLE = "originagenda_title"
FIELD_CONDITIONS = "conditions_fr"
FIELD_REGISTRATION = "registration"

# --------------------------------------------------------------------------
# Périmètre géographique : Bordeaux Métropole (28 communes)
# Source : INSEE — https://www.insee.fr/fr/metadonnees/geographie/intercommunalite/243300316-bordeaux-metropole
# Vérifié le 15/07/2026 (recoupé avec Wikipédia et 2 annuaires de communes indépendants).
# --------------------------------------------------------------------------
BORDEAUX_METROPOLE_COMMUNES = [
    "Ambarès-et-Lagrave",
    "Ambès",
    "Artigues-près-Bordeaux",
    "Bassens",
    "Bègles",
    "Blanquefort",
    "Bordeaux",
    "Bouliac",
    "Bruges",
    "Carbon-Blanc",
    "Cenon",
    "Eysines",
    "Floirac",
    "Gradignan",
    "Le Bouscat",
    "Le Haillan",
    "Le Taillan-Médoc",
    "Lormont",
    "Martignas-sur-Jalle",
    "Mérignac",
    "Parempuyre",
    "Pessac",
    "Saint-Aubin-de-Médoc",
    "Saint-Louis-de-Montferrand",
    "Saint-Médard-en-Jalles",
    "Saint-Vincent-de-Paul",
    "Talence",
    "Villenave-d'Ornon",
]

# --------------------------------------------------------------------------
# Fenêtre temporelle : uniquement les événements dont la première occurrence
# a commencé il y a moins de DAYS_HISTORY jours (couvre l'historique récent
# ET les événements à venir, dont la date de début est par définition future).
# --------------------------------------------------------------------------
DAYS_HISTORY = 365

# --------------------------------------------------------------------------
# Pagination / robustesse des appels API
# --------------------------------------------------------------------------
PAGE_SIZE = 100          # max autorisé par l'API Explore v2.1
MAX_PAGES = 100          # garde-fou PAR COMMUNE (pagination désormais partitionnée
                         # par commune, cf. fetch_raw_data.py) : 100 x 100 = 10 000
                         # événements max par commune, largement suffisant
REQUEST_TIMEOUT_SECONDS = 15
REQUEST_DELAY_SECONDS = 0.2   # pause entre deux pages, pour rester correct avec l'API publique
MAX_RETRIES = 3

# --------------------------------------------------------------------------
# Chemins de sortie
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DATA_FILE = RAW_DATA_DIR / "openagenda_bordeaux_metropole_raw.json"
RAW_META_FILE = RAW_DATA_DIR / "openagenda_bordeaux_metropole_meta.json"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "events.json"

# --------------------------------------------------------------------------
# Vectorisation (Mistral) + découpage en chunks
# --------------------------------------------------------------------------
load_dotenv()  # lit le fichier .env local (jamais commité, voir .gitignore)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

EMBEDDING_MODEL = "mistral-embed"          # 1024 dimensions

# Découpage en chunks (RecursiveCharacterTextSplitter) : valeurs choisies
# après inspection de la distribution réelle des longueurs de texte sur les
# 10 719 événements (16/07/2026) — médiane 746, 95e percentile 1961
# caractères. Avec 1500, ~11% des événements sont effectivement découpés en
# plusieurs chunks ; le reste (l'écrasante majorité) tient dans un seul.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

# Un appel `embed_documents(batch)` = 1 requête HTTP. Le compte Mistral de
# Thomas est plafonné à 1 requête/seconde (le vrai facteur limitant, pas les
# tokens/minute) : on boucle nous-mêmes par petits paquets avec une pause,
# plutôt que de faire confiance à un éventuel comportement interne de la
# librairie sur ce point (non documenté officiellement, cf. discussion).
EMBEDDING_BATCH_SIZE = 50
EMBEDDING_REQUEST_DELAY_SECONDS = 1.1
EMBEDDING_MAX_RETRIES = 3

# Sauvegarde intermédiaire : les vecteurs sont chers à calculer (appels API
# payants, ~plusieurs minutes) alors que l'indexation Faiss est gratuite et
# quasi instantanée (calcul local). En séparant les deux, on peut retravailler
# l'index Faiss (étape suivante) sans jamais recalculer les vecteurs.
CHUNKS_FILE = PROCESSED_DATA_DIR / "chunks.json"          # texte + métadonnées, un par chunk
EMBEDDINGS_FILE = PROCESSED_DATA_DIR / "embeddings.npy"   # matrice numpy (n_chunks, 1024), même ordre que CHUNKS_FILE

VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"           # index Faiss (étape 3)

# --------------------------------------------------------------------------
# Chatbot RAG (étape 4)
# --------------------------------------------------------------------------
# "-latest" plutôt qu'un nom de version daté : Mistral maintient cet alias
# pointé automatiquement vers la version la plus récente de la gamme, donc
# le code reste valide même quand Mistral sort une nouvelle version.
# mistral-small-latest n'est pas disponible sur le compte utilisé pour ce
# projet (vérifié sur la page des limites de La Plateforme) -> medium.
RAG_CHAT_MODEL = "mistral-medium-latest"
RAG_TEMPERATURE = 0.3   # peu de créativité : on veut des réponses factuelles, ancrées sur le contexte fourni

RAG_FETCH_K = 100        # nombre de chunks bruts récupérés dans Faiss avant dédoublonnage
                         # Passé de 20 à 100 le 27/07/2026 : la fusion lexicale ne peut
                         # re-classer QUE ce que Faiss remonte déjà dans ce lot -- elle ne
                         # fait jamais apparaître un événement absent du top RAG_FETCH_K.
                         # Or "metal" concerne ~0,25% des chunks (vérifié le 19/07/2026),
                         # soit une trentaine d'événements sur ~12 643 chunks -- largement
                         # plus que ne pouvait en voir un lot de 20. Recherche Faiss locale
                         # (pas d'appel API), coût quasi nul même à 100 ; RAG_MAX_EVENTS
                         # (10) reste inchangé, donc aucun impact sur le nombre de tokens
                         # envoyés à Mistral.
RAG_MAX_EVENTS = 10      # nombre d'événements uniques max montrés au modèle (après dédoublonnage)
RAG_MAX_HYBRID_EVENTS = 5  # sous-plafond : la recherche hybride (date/mots-clés exacts) ne
                            # peut jamais monopoliser tous les emplacements finaux -- au moins
                            # RAG_MAX_EVENTS - RAG_MAX_HYBRID_EVENTS emplacements restent
                            # garantis à la recherche sémantique classique.

# Un mot-clé présent sur plus de ce pourcentage d'événements est jugé trop
# générique pour servir de déclencheur à la recherche hybride (ex: "bordeaux"
# à 12,6%, "musique"/"concert" à ~3,5% -- n'importe quelle question sur un
# concert les déclencherait, sans rapport avec leur pertinence réelle).
# Vérifié empiriquement le 19/07/2026 sur les 12 643 chunks : "metal"/"métal"
# à 0,25% reste largement sous ce seuil, "musique"/"concert"/"bordeaux" sont
# tous largement au-dessus -- séparation nette, pas de zone grise observée.
RAG_KEYWORD_MAX_FREQUENCY = 0.01

# Mots-clés exclus manuellement du vocabulaire hybride, indépendamment de
# leur fréquence. Découverts le 27/07/2026 en testant "cours de bachata" :
# "cours" (10 occurrences, donc largement sous RAG_KEYWORD_MAX_FREQUENCY)
# faisait remonter "Cours de dessin" et "Cours de théâtre" comme résultats
# hybrides garantis, sans aucun rapport avec la question. Le filtre de
# fréquence seul ne suffit pas : un mot peut rester rare en occurrences
# absolues tout en étant non-discriminant, s'il est partagé par des
# catégories d'événements sans rapport entre elles. Principe de tri utilisé
# (voir diagnostic via `python -m src.analyse_vocabulaire`) : les mots qui
# décrivent le FORMAT d'un événement (cours, atelier, soirée, séance,
# concours, réunion...) sont non-discriminants par construction -- n'importe
# quel sujet peut se décliner en cours ou en soirée -- contrairement aux
# mots de CONTENU/GENRE (metal, bachata, dessin...), qui restent le signal
# utile. Les tags composés ("cours de dessin", "soiree cubaine") ne sont PAS
# concernés : c'est justement le mot de contenu qui les rend légitimes.
# Volontairement exclu de cette liste : les noms de lieux/quartiers (ex.
# "pin galant", "musee d'aquitaine") -- une question du type "qu'est-ce qui
# se passe au Pin Galant ?" est une demande légitime, les exclure casserait
# ce cas d'usage (même raisonnement que pour les événements passés, cf.
# discussion du 27/07/2026).
#
# Lot ajouté le 31/07/2026 suite à un diagnostic systématique (regroupement
# des mots-clés par radical singulier/pluriel) : RAG_KEYWORD_MAX_FREQUENCY
# filtre par CHAÎNE EXACTE, pas par concept -- une variante (souvent le
# pluriel) d'un mot déjà exclu au singulier peut passer sous le seuil,
# comptée séparément (la normalisation _normaliser gère la casse et les
# accents, pas les variantes singulier/pluriel). Découvert via 424 faux
# positifs sur "Des concerts de musique dans le mois qui vient ?" (mot
# fautif : "concerts"), puis généralisé en comparant TOUT le vocabulaire par
# radical naïf. Deux familles parmi les résultats :
#   - mots de FORMAT (evenement, exposition, conference, rencontre, visite,
#     sortie, spectacle) -- même logique que cours/atelier/soirée plus haut ;
#   - mots de GENRE/PUBLIC trop larges (art, culture, musique, film,
#     histoire, famille, jeunesse, senior) -- même logique que
#     musique/concert/bordeaux, déjà exclus par le seuil de fréquence lui-même.
# Volontairement PAS ajoutés malgré le même diagnostic : "balade(s)" et
# "jardin(s)" -- plus proches d'un lieu/thème précis (cf. "Pin Galant" plus
# haut) que d'un format générique ; à revoir seulement si un faux positif
# concret est observé à l'usage plutôt que par principe.
RAG_MOTS_CLES_EXCLUS = {
    "cours", "soiree", "soirees", "concours", "ateliers",
    "seance speciale", "reunion d'information", "activite", "activites",
    "animations", "quartier", "concerts",
    "evenement", "evenements", "expositions", "conferences",
    "rencontres", "visites", "sorties", "spectacles",
    "arts", "artes", "cultures", "musiques", "films", "histoires",
    "familles", "jeunesse", "jeunes", "jeune", "jeunesses", "jeune publics",
    "senior", "recrutements", "week-end", "faire", "en famille",
}
# "activites" (pluriel) ajouté le 31/07/2026 : "activite" (singulier) était
# déjà exclue manuellement depuis la découverte de "cours", mais son
# pluriel ne l'était pas -- angle mort du lot précédent, qui ne comparait
# que face au SEUIL AUTOMATIQUE (RAG_KEYWORD_MAX_FREQUENCY), jamais face à
# la liste d'exclusion manuelle elle-même. Explique Q5 et Q6 du jeu de test
# (toutes deux contiennent "activités").
#
# Repéré à la même occasion, et volontairement PAS ajouté : le regroupement
# naïf par radical rapproche à tort "cours" (classe) et "course"/"courses"
# (course à pied) -- mots sans rapport, malgré un radical commun une fois
# le "s"/"e" final retiré. Un futur audit du même type devra vérifier
# chaque paire à la main plutôt que faire confiance au seul radical.
#
# "week-end" ajouté le 31/07/2026, confirmé via
# _detecter_mots_cles_dans_question() sur Q12 : même famille que "cours" --
# un marqueur de MOMENT, pas de contenu, n'importe quel type d'événement
# peut avoir lieu "le week-end" (5 occurrences).
#
# "concert gratuit" (tag composé, matché sur la même question) envisagé
# puis volontairement écarté, sur décision de Thomas : seulement 2
# occurrences, impact jugé trop marginal pour justifier une entrée dédiée.
#
# "faire" et "en famille" ajoutés le 31/07/2026, confirmés via
# _detecter_mots_cles_dans_question() sur Q5 : "faire" (16 occurrences) est
# un verbe, aucun contenu informatif en soi ; "en famille" (5 occurrences)
# est une variante phrasée du même marqueur de public que "famille"/
# "familles", déjà exclus.

# En dessous de ce score de similarité cosinus (exact, calculé via le score
# BRUT d'un index MAX_INNER_PRODUCT sur des vecteurs normalisés — voir
# build_faiss_index.py et chatbot.py pour le détail), on considère qu'aucun
# événement du contexte n'est vraiment pertinent -> on ne consulte pas
# Mistral du tout et on renvoie un message direct (garde-fou anti-
# hallucination). Ce seuil ne s'applique QUE s'il n'y a aucun résultat
# hybride (date/mot-clé exact) parmi les sources — un résultat hybride est
# une preuve de pertinence en soi, indépendante de son score sémantique
# (souvent mauvais par construction, cf. discussion). Valeur réhaussée par
# rapport à la version précédente (0.2) suite à l'augmentation générale des
# scores après le passage à MAX_INNER_PRODUCT.
RAG_RELEVANCE_THRESHOLD = 0.5

# Poids du bonus lexical (TF-IDF sur titre+mots-clés) ajouté au score
# sémantique brut -- voir _appliquer_bonus_lexical() et _score_lexical()
# dans chatbot.py pour la formule complète. Décidé le 27/07/2026 pour
# départager des candidats aux scores sémantiques trop proches (ex: "Fête
# cuivrée" à 0,77 contre un concert de METAL à 0,74, alors que seul le
# second contient le mot cherché).
#
# Valeur de départ, À AJUSTER EMPIRIQUEMENT après quelques tests : un
# score_lexical fort (mot de la question présent dans le titre ET les
# mots-clés d'un événement, donc proche de 1 une fois normalisé par le poids
# IDF de la question) décale alors le score d'environ 0,15 -- de quoi
# renverser un écart du type Fête cuivrée/METAL (0,03) sans écraser
# complètement le signal sémantique.
#
# ATTENTION : ce bonus fait mécaniquement remonter le score affiché dans les
# sources ET celui comparé à RAG_RELEVANCE_THRESHOLD juste au-dessus (le
# garde-fou anti-hallucination) -- ce seuil, calibré à l'origine sur le
# score sémantique BRUT, mérite d'être revérifié empiriquement après
# quelques tests plutôt que supposé encore valide tel quel.
RAG_POIDS_BONUS_LEXICAL = 0.15

# Mots vides (stopwords) retirés avant le calcul du score lexical (voir
# _tokeniser() dans chatbot.py). Réintroduits le 27/07/2026 après un test
# ayant montré qu'un mot comme "de" n'a PAS un IDF assez proche de 0 pour
# être neutralisé automatiquement sur ce corpus -- il peut peser presque
# autant qu'un mot de contenu rare, et fausser le score en faveur
# d'événements qui ne partagent que ce mot de liaison avec la question
# (ex: "cours **de** dessin" contre une question "cours **de** bachata").
# Liste volontairement courte : uniquement les mots de liaison les plus
# fréquents, pas une liste NLP exhaustive.
RAG_MOTS_VIDES = {
    "de", "la", "le", "les", "un", "une", "des", "du", "et", "a",
    "au", "aux", "ce", "cette", "pour", "dans", "sur", "avec", "en",
}

RAG_NO_RESULTS_MESSAGE = (
    "Je n'ai trouvé aucun événement correspondant à votre demande dans la "
    "base actuelle (Bordeaux Métropole, événements de moins d'un an). "
    "N'hésitez pas à reformuler ou à préciser votre recherche."
)
