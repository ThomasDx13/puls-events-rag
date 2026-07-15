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
python test_environment.py
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

**Le premier `python test_environment.py` (ou toute première utilisation du
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

## 6. Structure du projet (mise en place, sera complétée aux étapes suivantes)

```
puls-events-rag/
├── venv/                 # environnement virtuel (non versionné)
├── src/                  # code du pipeline RAG (indexation, retrieval, chatbot)
├── data/
│   ├── raw/               # données brutes Open Agenda téléchargées
│   └── processed/         # données nettoyées/filtrées (< 1 an, périmètre géo)
├── vector_store/          # index Faiss généré (régénérable via script)
├── tests/                 # tests unitaires (dont validation date/périmètre)
├── notebooks/             # explorations ponctuelles
├── reports/               # rapport technique, présentation
├── requirements.txt
├── test_environment.py    # script de contrôle de l'environnement
└── README.md
```

## 7. Statut

- [x] Étape 1 — Environnement de développement (venv, dépendances, vérification)
- [ ] Étape 2 — Extraction et pré-traitement des données Open Agenda
- [ ] Étape 3 — Vectorisation et indexation Faiss
- [ ] Étape 4 — Pipeline RAG (retrieval + génération Mistral)
- [ ] Étape 5 — Jeu de test annoté (questions/réponses)
- [ ] Étape 6 — Tests unitaires
- [ ] Étape 7 — Rapport technique
- [ ] Étape 8 — Présentation PowerPoint
- [ ] Étape 9 — Démo live
