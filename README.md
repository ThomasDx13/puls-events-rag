# Puls-Events — POC Chatbot RAG (événements culturels)

POC d'un chatbot de recommandation d'événements culturels, augmenté par
récupération d'information (RAG) sur une base vectorielle construite à partir
des données publiques Open Agenda.

Stack : **LangChain** (orchestration) · **Mistral** (LLM + embeddings) ·
**Faiss** (base vectorielle, backend CPU).

---

## 1. Prérequis

- Python 3.10 ou supérieur (testé avec Python 3.12 et 3.14)
- pip
- Une clé API Mistral (à obtenir sur [console.mistral.ai](https://console.mistral.ai)) —
  nécessaire à partir de l'étape d'intégration du LLM, pas pour cette étape 1.

## 2. Installation de l'environnement

```bash
# 1. Cloner le dépôt
git clone <url-du-repo>
cd puls-events-rag

# 2. Créer l'environnement virtuel
python3 -m venv venv

# 3. L'activer
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 4. Installer les dépendances
pip install -r requirements.txt
```

## 3. Vérifier l'installation

Un script de contrôle est fourni pour s'assurer que toutes les bibliothèques
critiques (`langchain`, `langchain-mistralai`, `faiss`, `mistralai`) sont bien
importables :

```bash
python check_environment.py
```

Sortie attendue :

```
[OK]    langchain
[OK]    langchain_community
[OK]    langchain_mistralai
[OK]    faiss (CPU)
[OK]    mistralai
[OK]    pandas
Tous les imports ont réussi. L'environnement est prêt.
```

## 4. Dépendances principales

| Package             | Rôle                                                        |
|----------------------|--------------------------------------------------------------|
| `langchain`          | Orchestration du pipeline RAG (chaînes, prompts, retrievers) |
| `langchain-community`| Intégrations tierces pour LangChain (loaders, etc.)          |
| `langchain-mistralai`| Intégration officielle LangChain ↔ Mistral (LLM + embeddings)|
| `mistralai`           | Client Python officiel de l'API Mistral                     |
| `faiss-cpu`           | Base vectorielle (recherche de similarité), backend CPU     |
| `pandas`              | Manipulation des données Open Agenda avant vectorisation    |
| `python-dotenv`       | Chargement de la clé API Mistral depuis un fichier `.env`   |

Deux fichiers de dépendances, à deux niveaux :

- **`requirements.in`** — les dépendances *directes*, celles réellement
  importées dans le code (liste lisible, avec commentaires).
- **`requirements.txt`** — le fichier de *lock* (`pip freeze`), toutes les
  dépendances (directes + transitives) épinglées à une version exacte. C'est
  celui-ci qu'on installe (`pip install -r requirements.txt`) pour garantir
  que l'environnement est identique quelle que soit la machine.

Après ajout/modification d'une dépendance directe dans `requirements.in`,
regénérer le lock avec :

```bash
pip install -r requirements.in
pip freeze > requirements.txt
```

### ⚠️ Point de vigilance corrigé

La brief mentionnait `pip install ... mistral`. Ce nom de package **n'est pas
le bon** : le SDK officiel Mistral sur PyPI se nomme **`mistralai`**, et
l'intégration LangChain officielle s'appelle **`langchain-mistralai`**. C'est
ce qui a été installé ici.

### Note pour la suite

`langchain-community` affiche un avertissement de dépréciation (le mainteneur
recommande de migrer vers des packages d'intégration autonomes). Ça n'empêche
rien pour ce POC, mais si un loader spécifique de `langchain-community` est
utilisé plus tard, il faudra vérifier s'il existe une alternative dédiée.

## 5. Dépannage (Windows)

**Le premier `python check_environment.py` (ou toute première utilisation du
venv) semble bloqué / très lent, notamment sur l'import de `langchain_mistralai`
ou `httpx`.**

C'est l'antivirus (Windows Defender ou équivalent) qui scanne chaque nouveau
fichier du venv à son premier accès — un venv installe plusieurs milliers de
petits fichiers d'un coup. Une fois le scan fait, les exécutions suivantes
sont normales. Si c'est trop long ou bloque vraiment :
- Relancer simplement le script une seconde fois.
- Si ça persiste, exclure le dossier du projet du scan temps réel :
  Windows Sécurité → Protection contre les virus et menaces → Gérer les
  paramètres → Exclusions → Ajouter un dossier (le dossier du projet).

**Éviter aussi de placer le projet dans un dossier synchronisé par
OneDrive/Google Drive/Dropbox** (ex. `Bureau`, `Documents` si la synchronisation
est activée dessus) : un venv contient trop de petits fichiers pour ces
outils de synchronisation, ce qui peut causer des lenteurs ou blocages
similaires, voire des erreurs liées à la limite de longueur de chemin Windows.

## 6. Extraction, vectorisation et tests (étape 2)

### 6.1 Extraction et nettoyage

Périmètre : **Bordeaux Métropole** (28 communes), événements dont la première
occurrence a commencé il y a moins d'un an (couvre l'historique récent et les
événements à venir). Source : dataset public
[`evenements-publics-openagenda`](https://public.opendatasoft.com/explore/dataset/evenements-publics-openagenda)
sur Opendatasoft.

**1. Valider le schéma de l'API (à faire une fois si tu repars de zéro)**

```bash
python -m src.fetch_raw_data --discover
```

Affiche un événement brut complet avec tous ses champs. Les noms de champs
utilisés dans `src/config.py` (`FIELD_TITLE`, `FIELD_LOCATION_CITY`, etc.) ont
été confirmés le 16/07/2026 sur un enregistrement réel (voir le commentaire en
tête du bloc `FIELD_*` dans `config.py` pour le détail). Si tu modifies le
périmètre ou que l'API évolue, relance `--discover` pour vérifier que rien n'a
changé côté schéma avant de lancer une extraction complète.

**2. Extraction complète**

```bash
python -m src.fetch_raw_data
```

Récupère les événements du périmètre, **une commune à la fois** (28 requêtes
indépendantes, chacune paginée séparément) plutôt qu'une seule grosse requête
sur les 28 communes en même temps. Raison : l'API Opendatasoft plafonne
`offset + limit` à 10 000 résultats *par requête*, et Bordeaux Métropole
dépasse ce seuil à elle seule (~10 700 événements sur la période). Partitionner
par commune contourne la limite, chaque commune ayant un total individuel
largement en dessous.

Sauvegarde le brut dans `data/raw/openagenda_bordeaux_metropole_raw.json` +
les métadonnées de l'extraction (date, stratégie de pagination, décompte par
commune) dans `data/raw/openagenda_bordeaux_metropole_meta.json`.

**3. Nettoyage et structuration**

```bash
python -m src.preprocess
```

Charge le brut, **revalide indépendamment** la date et la ville de chaque
événement (ne fait pas confiance au filtre serveur), élimine les doublons et
les événements avec titre/description manquant, et sauvegarde le résultat
structuré dans `data/processed/events.json`. Un rapport (nombre d'événements
rejetés par motif) s'affiche à la fin.

**Résultat de l'extraction (16/07/2026)**

| | |
|---|---|
| Événements bruts récupérés (28 communes, < 1 an) | 10 718 |
| Rejetés — titre manquant | 3 |
| Rejetés — description manquante | 0 |
| Rejetés — date hors périmètre (revalidation) | 0 |
| Rejetés — ville hors périmètre (revalidation) | 0 |
| Rejetés — doublon | 0 |
| `timings` non parsable (information, non bloquant) | 0 |
| **Événements propres, prêts à indexer** | **10 715** |

Points notables :
- Les rejets `date_hors_perimetre` et `ville_hors_perimetre` à 0 confirment que
  le filtre serveur (dans `fetch_raw_data.py`) et la revalidation côté client
  (dans `preprocess.py`) sont cohérents entre eux — les deux mécanismes,
  indépendants, aboutissent au même résultat.
- Les 3 événements sans titre ont été inspectés manuellement : ce sont de
  vrais événements avec un champ `title_fr` simplement non renseigné par
  leur organisateur sur OpenAgenda (pas un bug de parsing côté script).
- Volume jugé suffisant pour le POC : pas besoin d'élargir le périmètre à la
  Gironde/région pour l'instant.

### 6.2 Champs supplémentaires (16-17/07/2026)

En complément du schéma initial, 3 champs de l'API sont désormais captés
dans `data/processed/events.json` :

- **`long_description`** — version texte de `longdescription_fr`, HTML retiré
  via BeautifulSoup (`bs4`) plutôt qu'une regex maison, plus robuste face à
  des balises réelles imbriquées (`<p><strong>...</strong><br>...</p>`).
- **`conditions`** — infos d'accès/tarif (ex. *"Gratuit sans réservation"*),
  texte simple.
- **`registration`** — moyens d'inscription/contact (lien, email, téléphone),
  parsé depuis une chaîne JSON en une vraie liste structurée. Jamais inclus
  dans le texte vectorisé (aucune valeur sémantique pour la recherche), mais
  conservé en métadonnée pour un usage futur par le chatbot.

### 6.3 Vectorisation (`src/vectorize.py`)

```bash
cp .env.example .env   # puis renseigner MISTRAL_API_KEY (et HF_TOKEN, optionnel — voir plus bas)
python -m src.vectorize --sample 20   # test rapide avant la totalité
python -m src.vectorize               # vectorisation complète
```

Étapes du script :

1. Pour chaque événement, un **header** (titre + description courte + lieu +
   date en français) est construit puis **préfixé à chaque chunk** — jamais
   isolé comme son propre chunk (voir "chunks orphelins" ci-dessous). La date
   est reformatée en français sans dépendre de la locale système (ex.
   *"mercredi 16 novembre 2026"*), pour éviter les soucis déjà rencontrés
   avec Windows sur ce projet.
2. Le **body** (description longue nettoyée + conditions + mots-clés) est
   découpé en chunks avec `RecursiveCharacterTextSplitter` (LangChain) :
   `chunk_size=1500`, `chunk_overlap=300` — valeurs choisies après inspection
   de la distribution réelle des longueurs de texte (médiane 746 caractères,
   95e percentile 1961, sur 10 719 événements). Avec ces valeurs, environ 11%
   des événements sont effectivement découpés en plusieurs morceaux ; le
   reste (grande majorité) tient en un seul chunk.
3. Chaque chunk est envoyé à l'API Mistral (modèle `mistral-embed`, 1024
   dimensions) par batchs de 50, avec une pause d'1,1s entre chaque appel
   pour respecter la limite du compte (**1 requête/seconde** — le vrai
   facteur limitant, pas les tokens/minute). En cas d'erreur possiblement
   liée à la taille du batch, celui-ci est automatiquement réduit et l'appel
   retenté.
4. Résultat sauvegardé en 2 fichiers alignés par position :
   `data/processed/chunks.json` (texte + métadonnées de chaque chunk) et
   `data/processed/embeddings.npy` (matrice numpy des vecteurs). Séparé
   volontairement de l'indexation Faiss (étape suivante) : la vectorisation
   coûte des appels API payants, l'indexation est gratuite et locale — pas
   besoin de repayer pour retravailler l'index plus tard.

**Chunks orphelins — problème rencontré et corrigé.** Une première version
séparait titre/description, description longue et lieu/date par des sauts de
paragraphe, découpés indépendamment par le splitter. Conséquence :
`RecursiveCharacterTextSplitter` isolait parfois le titre seul (ou le
lieu/date seul) comme un chunk à part entière d'à peine ~50 caractères, sans
aucun contexte exploitable. Corrigé en préfixant systématiquement le header à
chaque morceau du body plutôt que de le laisser comme section indépendante.

**Résultat de la vectorisation (17/07/2026)**

| | |
|---|---|
| Événements en entrée | 10 715 |
| Chunks générés | 12 643 |
| Dimension des vecteurs (`mistral-embed`) | 1024 |
| Erreurs de batch pendant l'exécution complète | 0 |

*Note technique* : `MistralAIEmbeddings` télécharge un tokenizer depuis
Hugging Face (pour estimer des tailles de batch en interne — sans lien avec
le calcul des vecteurs eux-mêmes, qui vient toujours de l'API Mistral). Un
`HF_TOKEN` gratuit ([huggingface.co](https://huggingface.co), permission
"Read" suffisante) évite un avertissement lié à la limite de débit anonyme de
Hugging Face — sans impact sur le résultat si on l'ignore.

### 6.4 Tests unitaires (date/périmètre)

Deux fichiers dans `tests/`, à deux niveaux différents :

- **`test_preprocess.py`** — teste `_is_recent_enough`/`_is_in_perimeter` de
  `preprocess.py` de façon isolée, avec des cas choisis à la main (dont les
  cas limites : pile à 365 jours, casse différente, espaces superflus, date
  malformée). Ne dépend d'aucune donnée réelle, peut tourner à tout moment
  (y compris juste après un `git clone`, avant toute extraction).
- **`test_data_quality.py`** — vérifie les VRAIES métadonnées de
  `data/processed/chunks.json` (date + ville de chaque chunk), tel que
  demandé par la consigne du projet. `SKIPPED` proprement si le pipeline n'a
  pas encore été lancé, plutôt que de planter.

```bash
python -m pytest tests/ -v
```

**Résultat (17/07/2026)** : tous les tests passent — 20/20 sur
`test_preprocess.py`, 4/4 sur `test_data_quality.py` (12 643 chunks, aucune
violation de date ou de périmètre détectée).

## 7. Indexation Faiss (étape 3)

```bash
python -m src.build_faiss_index
python -m pytest tests/test_faiss_index.py -v
```

**Construction (`src/build_faiss_index.py`)** : charge `chunks.json` +
`embeddings.npy` (déjà calculés par `vectorize.py`, aucun nouvel appel API
ici — étape gratuite et locale, relançable à volonté), construit l'index via
LangChain (`FAISS.from_embeddings`), vérifie que tous les événements sont
représentés, puis sauvegarde dans `vector_store/`.

**Décisions de conception :**

- **Index Flat (recherche exacte), pas IVF/HNSW.** À l'échelle de ce POC
  (~12 600 vecteurs), une recherche exacte reste de l'ordre de quelques
  millisecondes à quelques dizaines de ms (mesuré : voir résultats
  ci-dessous) — les index approximatifs (IVF, HNSW) deviennent intéressants
  à partir de dizaines/centaines de milliers de vecteurs, pas à ce volume.
  Les ajouter maintenant introduirait des paramètres à régler (nombre de
  clusters, taille de graphe...) sans bénéfice mesurable, avec une perte de
  précision pour rien. **Recommandation pour la version finale** si le
  volume d'événements grossit significativement (ex: extension à toute la
  région ou au national) : revisiter ce choix.
- **Distance : `MAX_INNER_PRODUCT`** (pas `COSINE`, changé le 19/07/2026 — voir
  7.1). Pour des vecteurs normalisés (norme 1, garanti par Mistral), le
  produit scalaire brut renvoyé par Faiss **est** exactement la similarité
  cosinus, sans aucune transformation intermédiaire — donc aucune formule
  approximative à appliquer, contrairement à `COSINE`.

### 7.1 Deux bugs Faiss découverts en creusant des scores incohérents (18-19/07/2026)

En creusant un cas où un événement pertinent (score cosinus réel 0.6970,
vérifié par calcul brute-force indépendant) n'apparaissait pas dans le top
100 alors que des résultats moins pertinents (0.67) y figuraient, deux bugs
distincts ont été trouvés et corrigés :

1. **`FAISS.save_local()`/`load_local()` ne persistent pas `distance_strategy`
   ni `normalize_L2`.** Seuls l'index Faiss brut et le magasin de documents
   sont sauvegardés — à chaque rechargement, ces réglages retombaient
   silencieusement sur leur valeur par défaut (`EUCLIDEAN_DISTANCE`). Ça ne
   faussait pas la **sélection** des résultats (la recherche brute par
   distance reste équivalente au classement cosinus pour des vecteurs
   normalisés), mais faussait le **score affiché** pour chacun (mauvaise
   formule de conversion). Fix : repréciser ces paramètres explicitement à
   chaque appel de `FAISS.load_local()` (voir `load_vectorstore()` dans
   `build_faiss_index.py`).
2. **La fonction de conversion "score de pertinence" de LangChain est
   approximative pour `COSINE`, et carrément inversée pour
   `MAX_INNER_PRODUCT`** (`1.0 - distance` au lieu de `1.0 - distance/2` pour
   `COSINE` ; testé empiriquement pour `MAX_INNER_PRODUCT` : un cosinus de
   1.0, la meilleure correspondance possible, donnait un score de 0.0).
   Fix : passer de `COSINE` à `MAX_INNER_PRODUCT` (index `IndexFlatIP`) et
   utiliser `similarity_search_with_score()` (score **brut**, jamais
   transformé) plutôt que `similarity_search_with_relevance_scores()`.

**Aucun de ces deux bugs n'a jamais affecté quels événements étaient
sélectionnés/recommandés** — uniquement le nombre affiché à côté. Vérifié à
chaque étape par calcul indépendant (produit scalaire numpy manuel comparé
au résultat Faiss).

**Avertissements rencontrés (sans impact, vérifiés)** :
- *"Normalizing L2 is not applicable for metric type: DistanceStrategy.COSINE"*
  — faux avertissement de `langchain-community` : la normalisation est bel
  et bien appliquée (vérifié en lisant le code source, et confirmé
  empiriquement par un score de 1.0000 exact sur un test avec un vecteur
  identique).
- Dépréciation de `langchain-community` (déjà notée à l'étape 1).

**Résultat (17/07/2026)**

| | |
|---|---|
| Vecteurs indexés | 12 643 |
| Événements représentés | 10 719 / 10 719 (complétude vérifiée) |
| Temps de recherche Faiss (mesuré, hors appel réseau Mistral) | 16,33 ms |

**Tests (`tests/test_faiss_index.py`)** — 6/6 passent :

| Test | Vérifie |
|---|---|
| `test_index_contient_autant_de_vecteurs_que_de_chunks` | Complétude (nombre de vecteurs = nombre de chunks) |
| `test_tous_les_evenements_sont_representes` | Aucun événement absent de l'index |
| `test_recherche_retrouve_le_bon_evenement` | Chercher le titre exact d'un événement le retrouve dans le top 5 |
| `test_recherche_par_mots_cles_retrouve_des_evenements_pertinents` | Une recherche thématique (mots-clés seuls, pas le titre) retrouve un résultat pertinent |
| `test_filtrage_par_date_exacte_fonctionne` | Le filtrage sur métadonnées (`filter=`) retrouve fiablement tous les événements d'un jour donné |
| `test_recherche_faiss_reste_rapide` | Recherche Faiss < 200 ms, mesurée isolément du temps réseau |

Contrairement aux autres tests du projet, ceux-ci font de vrais appels à
l'API Mistral (embedding de la requête de recherche) — un choix volontaire
pour valider le comportement réel plutôt que simulé, au prix d'un tout petit
coût API par lancement.

### 7.2 Limitations connues et observations (découvertes en testant sur les vraies données)

- **La recherche sémantique seule ne fiabilise pas les critères précis
  (dates, prix, identifiants).** Chercher une date exacte par similarité
  ramène des dates "du même genre" (même mois/saison), pas forcément la
  bonne — un embedding ne fait pas d'arithmétique sur les nombres. Solution
  retenue : filtrage strict sur métadonnées (`filter=`) en complément de la
  recherche vectorielle, pas à la place — généralisée aux mots-clés à
  l'étape 4 (voir section 8.4).
- **Piège LangChain sur `filter=`** : par défaut, le filtre ne s'applique
  qu'aux 20 voisins sémantiques les plus proches de la requête (`fetch_k=20`),
  pas à l'index entier — `k` ne contrôle que le nombre de résultats *gardés*
  après filtrage, pas le nombre de candidats *examinés* avant. Nécessite de
  fixer `fetch_k` explicitement au nombre total de chunks pour un filtrage
  fiable sur l'ensemble des données.
- **Les scores de similarité ne sont pas proportionnels à la quantité de
  texte qui correspond.** Chercher le titre exact d'un événement donne un
  score cosinus \~0,77 (pas 1,0, puisque le chunk indexé contient bien plus
  que le titre seul), et une recherche par mots-clés courts peut donner un
  score très proche — un texte court et thématiquement pur peut aligner
  aussi bien qu'un texte long. Pas un signe de dysfonctionnement.
- **La similarité sémantique seule ne suffit pas à bien classer des
  résultats déjà "dans le même thème général".** Cas vérifié : sur une
  recherche de concert de musique metal, les vrais concerts metal (score
  cosinus réel jusqu'à 0.70) étaient classés derrière des concerts d'autres
  genres (jusqu'à 0.77) — l'un d'eux (ASHEN) au rang réel 3106 sur 12 643,
  totalement hors de portée d'un `top_k` raisonnable. **Recommandation
  suivie dès l'étape 4** : recherche hybride (filtre exact en complément),
  voir section 8.4.
- **Doublons apparents dans les données source.** Un même événement ("Forum
  Mystère", 27/11/2025) apparaît sous 4 `uid` OpenAgenda différents,
  correspondant à 4 créneaux horaires (13h, 14h, 14h30, 15h30) publiés comme
  des événements séparés plutôt que regroupés dans un seul `timings` — un
  choix de l'organisateur/agenda source, pas un bug du pipeline. Notre
  déduplication (sur `uid` exact) ne peut pas détecter ce cas.
  **Recommandation pour la version finale** : dédupliquer aussi sur
  titre+adresse+jour si ce cas s'avère fréquent, et/ou regrouper ces
  créneaux à l'affichage côté chatbot plutôt qu'à l'indexation.
- **Incohérence de fuseau horaire entre deux champs de l'API.**
  `firstdate_begin` (→ notre `date_start`) est fourni en UTC, alors que
  `timings.begin/end` est fourni en heure locale française (`+01:00` hiver /
  `+02:00` été) — vérifié sur des cas réels. Impact négligeable pour ce
  projet (rare qu'un événement culturel commence pile autour de minuit), mais
  à garder en tête : le jour calendaire calculé depuis `date_start` (UTC)
  peut différer du jour "vécu" en France pour un événement démarrant très tôt
  le matin ou très tard le soir heure de Paris.

## 8. Chatbot RAG (étape 4)

```bash
python -m src.chat_cli
streamlit run streamlit_app.py
python -m pytest tests/test_chatbot_scenarios.py -v -s
```

### 8.1 Architecture — deux chaînes LCEL

`src/chatbot.py` construit le pipeline avec LangChain Expression Language
(l'opérateur `|`, qui chaîne des "Runnables" — voir le docstring en tête du
fichier pour une explication pédagogique complète des concepts) :

- **`retrieval_chain`** : question → `{"context": texte formaté, "sources":
  [(Document, score, est_hybride), ...]}`. Recherche Faiss + recherche
  hybride (8.4), dédoublonnage par événement.
- **`generation_chain`** : `{"question", "context"}` → `AIMessage` complet
  (pas juste le texte — nécessaire pour `.usage_metadata`, voir 8.5).
  `prompt | modèle`, sans `StrOutputParser()`.

`ask()` orchestre les deux et décide, en Python simple (pas dans la chaîne),
si le garde-fou anti-hallucination (8.3) doit empêcher l'appel à Mistral.

### 8.2 Modèle de génération

**`mistral-medium-latest`** — `mistral-small-latest` n'est pas disponible
sur le compte utilisé pour ce projet (vérifié sur la page des limites de La
Plateforme). Alias `-latest` plutôt qu'une version datée : reste valide
automatiquement quand Mistral sort une nouvelle version de cette gamme.
Température fixée à 0.3 (peu de créativité, réponses ancrées sur le contexte
fourni plutôt qu'improvisées).

**Piste identifiée pour la suite, non implémentée** : `mistral-small-2506`
est aussi disponible sur ce compte, avec des limites bien plus confortables
(2 250 000 tokens/min, 5 req/s contre 25 000/0,83 pour medium) — à évaluer
pour la génération principale ou pour un futur reranking LLM (voir 8.4).

### 8.3 Garde-fou anti-hallucination

Avant d'appeler Mistral, on vérifie qu'au moins un résultat est assez
pertinent — sinon on renvoie directement un message standard, sans appeler
le LLM. Logique révisée après plusieurs itérations :

```python
a_un_match_hybride = any(est_hybride for _, _, est_hybride in sources)
meilleur_score_semantique = max(score pour les résultats NON hybrides, défaut 0.0)

if not a_un_match_hybride and meilleur_score_semantique < RAG_RELEVANCE_THRESHOLD:
    # message standard, Mistral jamais appelé
```

**Pourquoi cette forme précise** : un test réel (question hors-sujet
"Quelle est la capitale de la France ?") a montré qu'**aucun seuil
numérique ne sépare fiablement le pertinent du hors-sujet** sur ce
dataset — des questions hors-sujet ont parfois un meilleur score sémantique
que de vraies questions pertinentes (des quiz de culture générale
scoraient plus haut qu'un concert metal réellement présent dans la base).
Le seuil (`RAG_RELEVANCE_THRESHOLD = 0.5`) ne sert donc que de plancher pour
les questions **sans aucun résultat hybride** — le vrai filtrage sémantique
fin est délégué au prompt (voir règles dans `SYSTEM_PROMPT`), qui s'est
montré fiable sur tous les scénarios testés. Un résultat hybride, lui,
contourne complètement ce seuil : une correspondance exacte de mot-clé/date
est une preuve de pertinence en soi, même si son score sémantique est
mauvais (ce qui est fréquent et attendu pour ce type de résultat).

### 8.4 Recherche hybride (mots-clés et dates exacts, en complément du sémantique)

**Motivation** (section 7.2) : la recherche purement sémantique peut classer
un événement réellement pertinent très loin (rang 3106/12 643 observé), donc
hors de portée du contexte montré à Mistral, même si son score cosinus réel
est correct. Solution : généraliser le filtrage exact par métadonnées (déjà
utilisé pour les dates dans les tests de l'étape 3) et l'intégrer réellement
au chatbot — pas seulement aux tests, comme c'était le cas dans une première
version.

**Mécanisme** :
1. **Vocabulaire de mots-clés** construit une fois à l'import (lecture de
   `chunks.json`), **filtré par fréquence** (`RAG_KEYWORD_MAX_FREQUENCY =
   0.01`) : un mot-clé présent sur plus de 1% des événements est jugé trop
   générique pour servir de déclencheur. Vérifié empiriquement le
   19/07/2026 : "bordeaux" (12,6%), "musique"/"concert" (~3,5%) contre
   "metal"/"métal" (0,25%) — séparation nette, sans mot-clé ambigu identifié
   dans la zone intermédiaire.
2. **Détection dans la question** : mots-clés connus (comparaison par
   frontière de mot) + dates au format "JJ mois \[AAAA\]" en français (sans
   année, suppose la prochaine occurrence à venir).
3. **Normalisation casse + accents** (`unicodedata`, NFKD) appliquée à
   3 endroits (vocabulaire, question, **et** mots-clés bruts de chaque
   événement au moment du filtre — ce dernier point corrigeait un bug réel :
   l'événement ASHEN avait le mot-clé `"Metal"` en majuscule, jamais reconnu
   par une comparaison de chaînes sensible à la casse). `Metal`, `METAL`,
   `métal`, `MÉTAL`... convergent tous vers `metal`.
4. **Filtre exact sur l'index entier** (`fetch_k` = nombre total de chunks,
   pas seulement le voisinage sémantique — même piège que 7.2).
5. **Sous-plafond** `RAG_MAX_HYBRID_EVENTS = 5` (sur `RAG_MAX_EVENTS = 10`) :
   empêche un mot-clé très présent de monopoliser tous les emplacements
   finaux, garantit toujours de la place à la recherche sémantique classique.
6. **Priorité, pas tri par score** : les résultats hybrides sont placés en
   tête de liste avant dédoublonnage (pas triés avec les résultats
   sémantiques par score — leur score est souvent mauvais par construction,
   les trier ensemble les aurait exclus du top final).
7. **Exclusion manuelle complémentaire au filtre de fréquence**
   (`RAG_MOTS_CLES_EXCLUS`, config.py). Découvert le 27/07/2026 en testant
   "cours de bachata" : le mot-clé `"cours"` (10 occurrences sur 12 414
   chunks, donc largement sous le seuil de 1%) faisait remonter "Cours de
   dessin" et "Cours de théâtre" comme résultats hybrides garantis, sans
   rapport avec la question. Un mot peut rester rare en occurrences absolues
   tout en étant non-discriminant, s'il est partagé par des catégories
   d'événements sans rapport entre elles — le filtre de fréquence seul ne
   suffit pas à le détecter. Diagnostic outillé : `src/analyse_vocabulaire.py`
   (affiche le vocabulaire hybride actuel trié par fréquence, pour repérer
   ce type de mot visuellement plutôt qu'à l'aveugle). Principe de sélection
   retenu : les mots qui décrivent le FORMAT d'un événement (cours, atelier,
   soirée, séance, concours, réunion...) sont non-discriminants par
   construction — n'importe quel sujet peut se décliner en cours ou en
   soirée — contrairement aux mots de CONTENU/GENRE (metal, bachata,
   dessin...), qui restent le signal utile. **Volontairement exclus de cette
   liste** : les noms de lieux/quartiers (ex. "pin galant", "musée
   d'aquitaine") — une question du type "qu'est-ce qui se passe au Pin
   Galant ?" est une demande légitime, les exclure casserait ce cas d'usage
   (même raisonnement que pour les événements passés, voir 8.7).

**Résultat vérifié** (19/07/2026, question "Un concert de musique métal à
Bordeaux ?") : les 5 vrais concerts metal de la base (ASHEN, SOIRÉE METAL,
Slipknot tribute, ANTHRAX, IGORRR, ULTRA VOMIT) sont désormais tous détectés
et correctement recommandés par Mistral, qui ignore sans exception les
résultats "bruit" (musique classique, fêtes...) toujours présents dans le
contexte via les 5 emplacements sémantiques restants.

### 8.5 Suivi de la consommation de tokens

`ask()` retourne `usage: {"input_tokens", "output_tokens", "total_tokens"}`
(extrait de `AIMessage.usage_metadata`, d'où l'absence de `StrOutputParser()`
dans `generation_chain` — un parseur de sortie aurait réduit la réponse au
texte seul et perdu cette métadonnée). Affiché après chaque échange dans
`chat_cli.py` : utile vu la limite serrée du compte (25 000 tokens/min sur
`mistral-medium-latest`).

### 8.6 Tests (`tests/test_chatbot_scenarios.py`)

Contrairement aux tests précédents, la justesse d'une réponse en langage
naturel n'est pas vérifiable par un simple `assert` — pas de jeu de
questions/réponses annoté pour comparer (ce sera l'objet de l'étape 5). Ce
fichier vérifie donc ce qui EST automatisable (réponse non vide, sources
trouvées, garde-fou qui se déclenche bien sur une question hors-sujet) et
affiche le reste (`-s`) pour relecture humaine.

### 8.7 Distinction événements passés / à venir

**Bug rencontré (23/07/2026)** : sur une question type "un concert de metal
ce mois-ci ?", un événement passé (28/03/2026) a été présenté par Mistral
comme "le prochain concert" — alors que la date figurait correctement dans
le contexte fourni. Cause : un LLM n'a pas d'horloge, chaque appel API est
sans état ; sans qu'on lui donne explicitement la date du jour, Mistral n'a
aucune donnée pour comparer une date d'événement à "maintenant".

**Fix (23/07/2026)** : `{date_du_jour}` injecté dans `SYSTEM_PROMPT`, avec
une règle de comparaison explicite. Calculé via `date.today().isoformat()`
à **chaque appel** de `ask()` (pas au chargement du module) — le process
Streamlit/CLI peut tourner plusieurs jours, une date figée à l'import serait
vite obsolète.

**Décision de conception importante** : le retrieval ne filtre JAMAIS par
date. Un événement passé reste une réponse légitime à une question qui
porte explicitement dessus (ex: "quel groupe jouait au bar X la semaine
dernière ?") — un filtre "exclure le passé" côté recherche casserait ce cas
d'usage. La distinction passé/à venir est donc entièrement déléguée au
raisonnement de Mistral à la génération, pas au retrieval.

**Affiné le 27/07/2026** : ajout d'une règle pour mentionner la prochaine
occurrence à venir du même sujet quand rien ne correspond exactement à la
période demandée (plutôt que de répondre "rien trouvé" alors qu'un
événement pertinent existe plus tard dans le contexte). Un premier essai
a mélangé un événement passé et un événement à venir sous un même intitulé
("Le prochain concert est : [...ANTHRAX, déjà passé...] [...concert du
14/11...]") — trompeur bien que chaque ligne soit individuellement correcte.
Corrigé en exigeant des intitulés explicitement distincts ("le dernier en
date était..." / "le prochain est..."), jamais regroupés sous un même
en-tête ou une même liste.

### 8.8 Fusion score sémantique + score lexical (TF-IDF)

**Motivation** : le score sémantique seul se tasse parfois entre événements
pertinents et non-pertinents (ex. observé le 25/07/2026 : "Fête cuivrée",
un concert de cuivres sans rapport, à 0,77 contre un vrai concert de METAL
à 0,74). Le reranking LLM, déjà écarté pour coût (voir 8.10), n'est pas la
seule option : un score lexical déterministe, sans appel API, peut
compléter le signal sémantique.

**Formule**, appliquée à titre + mots-clés de chaque événement (pas la
description complète — même raison que pour l'embedding en 6.3 : trop de
texte non-discriminant dilue le signal) :

```
score_lexical = Σ TF(mot, évènement) × IDF(mot)   [mots communs question/évènement]
                ─────────────────────────────────────────────────────────
                Σ IDF(mot)                         [mots de la question]

IDF(mot) = log(N_évènements_uniques / df(mot))
TF(mot, évènement) ∈ {0, 1, 2} : +1 si présent dans le titre, +1 si présent
                                  dans les mots-clés (pas un compte brut de
                                  répétitions — titres/mots-clés trop courts
                                  pour que ce soit un signal fiable)

score_final = score_sémantique + RAG_POIDS_BONUS_LEXICAL × score_lexical
```

Remplace le score stocké dans les tuples `(doc, score, est_hybride)` —
décision assumée : un seul score cohérent dans tout le pipeline (garde-fou,
tri, affichage) plutôt que deux scores distincts. **Implication à
surveiller** : `RAG_RELEVANCE_THRESHOLD` (8.3) a été calibré sur le score
sémantique BRUT ; le bonus le fait mécaniquement remonter — un réajustement
empirique n'a pas encore été fait, à vérifier à l'usage.

Tokenisation MOT PAR MOT (via `_tokeniser()`), différente de 8.4 qui compare
des mots-clés ENTIERS — ici chaque mot est pondéré séparément, pour capter
un chevauchement partiel de vocabulaire que le filtre hybride exact ne
verrait pas du tout. `df(mot)` compté par événement UNIQUE (dédoublonné par
`uid`), pas par chunk (contrairement à 8.4 — corrigé dès l'écriture de ce
code neuf, l'ancien choix n'a pas été retouché).

**Leçon apprise en testant (27/07/2026), à noter honnêtement** : une
première version sans liste de mots vides (stopwords) supposait que l'IDF
suffirait à neutraliser les mots de liaison ("de", "la"...) automatiquement.
Faux sur ce corpus : un test synthétique a montré que "de" garde un IDF non
négligeable (apparaît dans ~50% d'un petit échantillon de titres "cours de
X"), suffisant pour faire gagner "Cours de dessin" contre "Salbazouk" sur
une question "cours de bachata" — l'inverse de l'effet recherché. Une petite
liste de mots vides (`RAG_MOTS_VIDES`, config.py) a été réintroduite après
ce test.

**`RAG_FETCH_K` 20 → 100 (même date)** : la fusion ne peut re-classer QUE ce
que Faiss remonte déjà dans le lot brut — elle ne fait jamais apparaître un
événement absent de ce lot. "metal" concernant ~0,25% des chunks (19/07),
soit une trentaine d'événements sur ~12 643, un lot de 20 était trop court
pour tous les voir. Recherche Faiss locale (pas d'appel API), coût quasi nul
même à 100 ; `RAG_MAX_EVENTS` (10) inchangé, aucun impact sur les tokens
envoyés à Mistral.

**Résultats vérifiés (27/07/2026)** :
- "cours de bachata" : "Cours de dessin"/"Cours de théâtre" disparaissent
  des résultats sémantiques, Salbazouk (vrai match bachata) score au-dessus
- "Un concert de musique métal à Bordeaux ce mois-ci ?" : le score final
  passe à 1,040 pour un vrai concert metal contre 0,832 pour "Fête cuivrée"
  (avec les scores sémantiques bruts réels 0,74/0,77) — l'ordre s'inverse
  comme attendu

**Limite ouverte, non résolue à ce stade** : sur la même question metal,
les résultats *purement sémantiques* (hors hybride) restent dominés par des
événements dont le titre contient "concert"/"musique" au sens large (déjà
identifiés le 19/07 comme trop génériques pour le filtre hybride, mais la
table IDF de la fusion lexicale ne "connaît" pas cette leçon-là — deux
mécanismes de généricité distincts dans le même fichier). Cause précise non
confirmée : soit ces mots ont un IDF trop faible pour être neutralisés,
soit le lot brut Faiss (même élargi à 100) ne contient simplement aucun
autre événement metal à ce rang. Test de diagnostic proposé
(`RAG_POIDS_BONUS_LEXICAL = 0` temporaire) pas encore effectué.

### 8.10 Limitations connues et pistes pour la version finale

- **Reranking LLM envisagé, écarté pour ce POC.** `mistralai-search-toolkit`
  propose un `LLMReRanker` (un modèle de chat reprompté pour juger la
  pertinence de chaque candidat individuellement) — mais son coût ("1 appel
  LLM par chunk" selon la doc officielle) le rendait irréaliste sur
  `mistral-medium-latest` (0,83 req/s). Redevient budgétairement réaliste
  avec `mistral-small-2506` (5 req/s), mais écarté du scope de ce POC malgré
  tout (décision du 27/07/2026). Une alternative sans appel API — fusion
  score sémantique + score lexical TF-IDF — a été implémentée à la place,
  voir 8.8 ; elle améliore le classement sans le résoudre complètement (voir
  limite ouverte dans cette même section). Un `CrossEncoderReRanker` existe
  aussi dans le même toolkit, mais nécessite d'héberger soi-même un modèle
  cross-encoder — hors scope pour ce POC.
- **La recherche hybride ne couvre que mots-clés et dates.** D'autres
  critères précis (prix exact, tranche d'âge, accessibilité PMR...) restent
  soumis aux limites de la recherche sémantique pure si un jour ils
  s'avèrent nécessaires.
- **Détection de date limitée au format explicite "JJ mois \[AAAA\]".** Les
  dates relatives ("ce week-end", "demain", "en novembre" sans jour précis)
  ne sont pas gérées — hors scope pour ce POC (cf. discussion), à couvrir
  si le besoin se confirme à l'usage.
- **Vocabulaire de mots-clés (et sa liste d'exclusion, 8.4) figés à l'import
  du module.** Si `vectorize.py` est relancé (nouvelles données) sans
  redémarrer le chatbot, le vocabulaire de déclenchement de la recherche
  hybride — et la table IDF de la fusion lexicale (8.8), même limitation —
  restent ceux de l'ancienne extraction — cas jugé trop marginal pour la
  complexité d'un rafraîchissement dynamique dans un POC.
- **`RAG_RELEVANCE_THRESHOLD` (8.3) calibré sur le score sémantique BRUT,
  pas encore réajusté depuis l'ajout du bonus lexical (8.8, 27/07/2026).**
  Le score comparé à ce seuil inclut désormais le bonus, qui le fait
  mécaniquement remonter — un réajustement empirique reste à faire.
- **Score sémantique encore imparfait sur du vocabulaire générique partagé
  ("concert", "musique"), même après la fusion lexicale (27/07/2026).** Sur
  une question metal, les résultats purement sémantiques restent dominés
  par des événements dont le titre contient ces mots au sens large — la
  table IDF de la fusion (8.8) ne "connaît" pas la leçon du 19/07 sur ces
  mots-clés trop génériques (deux mécanismes de généricité distincts dans
  le même fichier). Cause précise non confirmée à ce stade — diagnostic
  proposé (`RAG_POIDS_BONUS_LEXICAL = 0` temporaire) pas encore effectué.

