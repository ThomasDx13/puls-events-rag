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

