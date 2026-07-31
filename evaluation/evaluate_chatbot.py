"""
Évaluation du chatbot sur un jeu de questions/réponses annotées
(QA_annotees.json), pour mesurer la qualité des réponses par rapport à des
réponses de référence — consigne du mail du manager :

    "Pour évaluer le système, nous aurons besoin de créer un jeu de
    données test annoté de questions / réponses. Ce jeu test sera
    utilisé pour mesurer la qualité des réponses par rapport aux
    réponses annotées."

Ce n'est PAS un test pytest : il ne s'agit pas d'un pass/fail binaire mais
d'une mesure à interpréter. Contrairement à tests/test_chatbot_scenarios.py
(qui vérifie que le chatbot ne plante pas), ce script mesure la qualité
réelle des réponses générées.

Trois familles de questions dans QA_annotees.json (champ "type"), trois
façons de les évaluer :

- "hors_sujet" : comparaison stricte à config.RAG_NO_RESULTS_MESSAGE (le
    garde-fou anti-hallucination doit intercepter la question avant tout
    appel de génération). Résultat rapporté, jamais présupposé vrai : le
    seuil de pertinence est un heuristique imparfait (cas réel déjà
    observé où une question de trivia passait au-dessus du seuil).

- "generatif" : pas d'égalité stricte possible en langage naturel. Deux
    mesures complémentaires, qui ne capturent pas les mêmes erreurs :
      - similarité cosinus entre les embeddings (mistral-embed) de la
        réponse de référence et de la réponse générée — même logique que
        la recherche Faiss du projet ;
      - notation par un LLM-juge (mistral-medium-latest) qui compare les
        deux réponses sur le fond factuel (mêmes événements, dates,
        lieux), pas sur le style.
    Les deux réponses complètes sont toujours affichées, y compris pour
    les cas pièges (Q4, Q11, Q12) où le score seul ne dit pas si la
    réponse a correctement géré une absence de résultat ou une
    correspondance partielle plutôt que de forcer une suggestion
    approximative.

- "vague" : aucune mesure automatique pertinente (pas de bonne réponse
    objective) — affiché pour relecture humaine, comme
    test_question_vague dans l'ancien script de scénarios.

Usage :
    python -m evaluation.evaluate_chatbot
    python -m evaluation.evaluate_chatbot --questions chemin/vers/fichier.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from src import config
from src.chatbot import _statut_temporel, ask

FICHIER_QUESTIONS_DEFAUT = Path(__file__).resolve().parent / "QA_annotees.json"
DOSSIER_RAPPORTS = Path(__file__).resolve().parent / "rapports"

PROMPT_JUGE = """Tu es un évaluateur qualité pour un chatbot de recommandation \
d'événements culturels à Bordeaux Métropole.

On te donne une question, une réponse de référence (rédigée à partir des \
événements réellement présents dans la base) et une réponse générée par le \
chatbot à évaluer.

Note de 0 à 5 dans quelle mesure la réponse générée couvre les mêmes faits \
que la réponse de référence (mêmes événements, mêmes dates, mêmes lieux, \
même honnêteté quand aucun résultat ne correspond). Ignore le style \
d'écriture, seul le contenu factuel compte.

Réponds UNIQUEMENT avec un objet JSON strict, sans texte avant ni après, \
sans balises markdown, de la forme :
{{"score": <entier 0-5>, "justification": "<une phrase>"}}

Question : {question}

Réponse de référence :
{reference}

Réponse générée par le chatbot :
{generee}"""


def _verifier_pipeline_disponible():
    # Pas de vérification de MISTRAL_API_KEY ici : `from src.chatbot import ask`
    # (import en tête de ce fichier) lève déjà une RuntimeError si elle est
    # absente, avant même que cette fonction ne soit appelée -- la revérifier
    # ici serait du code mort.
    if not (config.VECTOR_STORE_DIR / "index.faiss").exists():
        sys.exit(
            "Index Faiss introuvable — lance d'abord tout le pipeline "
            "(fetch_raw_data, preprocess, vectorize, build_faiss_index)."
        )


def charger_questions(chemin: Path) -> list[dict]:
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(vecteur_a: list[float], vecteur_b: list[float]) -> float:
    a, b = np.array(vecteur_a), np.array(vecteur_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _formater_sources(sources: list[tuple]) -> str:
    """Formate la liste (Document, score, est_hybride) renvoyée par ask()
    en texte lisible pour le rapport -- pour pouvoir juger si un mauvais
    score vient d'une mauvaise récupération (les bons événements n'ont pas
    été trouvés) ou d'une mauvaise génération (les bons événements étaient
    là, mais mal exploités dans la réponse).

    Affiche aussi la date de fin (quand elle diffère de la date de début)
    et le statut temporel calculé par _statut_temporel (même fonction que
    celle utilisée par le chatbot lui-même dans _formater_contexte) --
    sans ça, impossible de vérifier depuis ce rapport si le filtre de
    période (cf. chatbot.py, _chevauche_periode) sélectionne des
    évènements réellement en chevauchement avec la période demandée, ou
    matche à tort sur autre chose (mot-clé, score sémantique)."""
    if not sources:
        return "(aucune source)"
    lignes = []
    for doc, score, est_hybride in sources:
        m = doc.metadata
        marqueur = " [hybride]" if est_hybride else ""
        statut = _statut_temporel(m)
        statut_str = f" [{statut}]" if statut else ""
        debut = (m.get("date_start") or "?")[:10]
        fin = (m.get("date_end") or "")[:10]
        date_str = f"{debut} (jusqu'au {fin})" if fin and fin != debut else debut
        lignes.append(
            f"- {m.get('title', '?')}{statut_str} ({date_str}, {m.get('city', '?')}) — score={score:.2f}{marqueur}"
        )
    return "\n".join(lignes)


def evaluer_hors_sujet(question: dict) -> dict:
    """Vérifie que le garde-fou anti-hallucination a bien intercepté la
    question avant tout appel de génération. Ne présuppose pas le résultat :
    le rapporte, qu'il soit positif ou négatif."""
    resultat = ask(question["question"])

    garde_fou_ok = (
        resultat["answer"] == config.RAG_NO_RESULTS_MESSAGE
        and resultat["usage"] is None
    )

    return {
        "id": question["id"],
        "type": "hors_sujet",
        "question": question["question"],
        "reponse_generee": resultat["answer"],
        "garde_fou_declenche": garde_fou_ok,
        "usage": resultat["usage"],
        "sources": resultat["sources"],
    }


def evaluer_generatif(question: dict, embeddings: MistralAIEmbeddings, juge: ChatMistralAI) -> dict:
    """Calcule les deux mesures (cosinus + LLM-juge) pour une question dont
    la référence a été écrite à la main à partir des données réelles."""
    resultat = ask(question["question"])
    reference = question["reponse_reference"]
    generee = resultat["answer"]

    vecteurs = embeddings.embed_documents([reference, generee])
    similarite = cosine_similarity(vecteurs[0], vecteurs[1])

    reponse_juge = juge.invoke(
        PROMPT_JUGE.format(question=question["question"], reference=reference, generee=generee)
    )
    texte_juge = reponse_juge.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        jugement = json.loads(texte_juge)
    except json.JSONDecodeError:
        jugement = {"score": None, "justification": f"Réponse du juge non parsable : {texte_juge!r}"}

    return {
        "id": question["id"],
        "type": question["type"],
        "question": question["question"],
        "reponse_reference": reference,
        "reponse_generee": generee,
        "similarite_cosinus": similarite,
        "score_llm_juge": jugement.get("score"),
        "justification_llm_juge": jugement.get("justification"),
        "notes": question.get("notes"),
        "sources": resultat["sources"],
    }


def evaluer_vague(question: dict) -> dict:
    """Aucune mesure automatique pertinente : affiché pour relecture
    humaine uniquement (pas de bonne réponse objective à cette question)."""
    resultat = ask(question["question"])
    return {
        "id": question["id"],
        "type": "vague",
        "question": question["question"],
        "reponse_generee": resultat["answer"],
        "nombre_sources": len(resultat["sources"]),
        "sources": resultat["sources"],
    }


def evaluer_tout(questions: list[dict]) -> list[dict]:
    embeddings = MistralAIEmbeddings(model=config.EMBEDDING_MODEL, mistral_api_key=config.MISTRAL_API_KEY)
    # Même modèle que la génération (config.RAG_CHAT_MODEL), mais température à 0 :
    # contrairement à RAG_TEMPERATURE=0.3 (réponses factuelles mais naturelles), on
    # veut ici une notation la plus reproductible possible, pas de variation de style.
    juge = ChatMistralAI(model=config.RAG_CHAT_MODEL, mistral_api_key=config.MISTRAL_API_KEY, temperature=0)

    resultats = []
    for question in questions:
        print(f"  [{question['id']}] {question['question']}")
        if question["type"] == "hors_sujet":
            resultats.append(evaluer_hors_sujet(question))
        elif question["type"] == "generatif":
            resultats.append(evaluer_generatif(question, embeddings, juge))
        elif question["type"] == "vague":
            resultats.append(evaluer_vague(question))
        else:
            raise ValueError(f"Type de question inconnu : {question['type']!r}")
        time.sleep(1)  # marge de sécurité, compte Mistral à limites serrées

    return resultats


def _moyennes_generatif(resultats: list[dict]) -> tuple[float | None, float | None]:
    """Moyennes cosinus/LLM-juge sur les questions 'generatif' -- factorisé
    car utilisé à la fois par le résumé console et le tableau récapitulatif
    du rapport Markdown (voir _construire_tableau_resume_markdown)."""
    generatifs = [r for r in resultats if r["type"] == "generatif"]
    if not generatifs:
        return None, None
    moy_cosinus = sum(r["similarite_cosinus"] for r in generatifs) / len(generatifs)
    scores_valides = [r["score_llm_juge"] for r in generatifs if r["score_llm_juge"] is not None]
    moy_juge = sum(scores_valides) / len(scores_valides) if scores_valides else None
    return moy_cosinus, moy_juge


def afficher_resume_console(resultats: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    for r in resultats:
        if r["type"] == "hors_sujet":
            statut = "OK" if r["garde_fou_declenche"] else f"ÉCHEC (garde-fou non déclenché, {len(r['sources'])} source(s))"
            print(f"{r['id']:5} [hors_sujet] {statut}")
        elif r["type"] == "generatif":
            score = r["score_llm_juge"]
            score_str = f"{score}/5" if score is not None else "N/A"
            print(f"{r['id']:5} [generatif]  cosinus={r['similarite_cosinus']:.2f}  juge={score_str}")
        else:
            print(f"{r['id']:5} [vague]      {r['nombre_sources']} source(s), à relire à l'oeil")

    moy_cosinus, moy_juge = _moyennes_generatif(resultats)
    if moy_cosinus is not None:
        print(f"\nMoyenne cosinus (questions generatif) : {moy_cosinus:.2f}")
        if moy_juge is not None:
            print(f"Moyenne score LLM-juge (questions generatif) : {moy_juge:.2f}/5")


def _construire_tableau_resume_markdown(resultats: list[dict]) -> str:
    """Tableau récapitulatif inséré en tête du rapport -- pour voir l'état
    d'ensemble en un coup d'oeil, sans avoir à faire défiler tout le
    document. Reprend le même contenu que afficher_resume_console(), en
    Markdown plutôt qu'en texte brut."""
    lignes = ["## Résumé\n", "| ID | Type | Résultat |", "|---|---|---|"]
    for r in resultats:
        if r["type"] == "hors_sujet":
            statut = "✅ OK" if r["garde_fou_declenche"] else f"❌ garde-fou non déclenché ({len(r['sources'])} source(s))"
            lignes.append(f"| {r['id']} | hors_sujet | {statut} |")
        elif r["type"] == "generatif":
            score = r["score_llm_juge"]
            score_str = f"{score}/5" if score is not None else "N/A"
            lignes.append(f"| {r['id']} | generatif | cosinus={r['similarite_cosinus']:.2f}, juge={score_str} |")
        else:
            lignes.append(f"| {r['id']} | vague | {r['nombre_sources']} source(s), à relire à l'oeil |")

    moy_cosinus, moy_juge = _moyennes_generatif(resultats)
    if moy_cosinus is not None:
        lignes.append("")
        lignes.append(f"Moyenne cosinus (questions generatif) : {moy_cosinus:.2f}  ")
        if moy_juge is not None:
            lignes.append(f"Moyenne score LLM-juge (questions generatif) : {moy_juge:.2f}/5")

    return "\n".join(lignes)


def sauvegarder_rapport_markdown(resultats: list[dict]) -> Path:
    DOSSIER_RAPPORTS.mkdir(exist_ok=True)
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    chemin_rapport = DOSSIER_RAPPORTS / f"rapport_{horodatage}.md"

    lignes = [
        f"# Rapport d'évaluation — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        _construire_tableau_resume_markdown(resultats),
        "\n---\n",
    ]
    for r in resultats:
        lignes.append(f"## {r['id']} — [{r['type']}]\n")
        lignes.append(f"**Question :** {r['question']}\n")

        if r["type"] == "hors_sujet":
            statut = "✅ garde-fou déclenché" if r["garde_fou_declenche"] else "❌ garde-fou NON déclenché"
            lignes.append(f"**Résultat :** {statut}\n")
            lignes.append(f"**Réponse générée :**\n> {r['reponse_generee']}\n")
            lignes.append(f"**Sources utilisées ({len(r['sources'])}) :**\n{_formater_sources(r['sources'])}\n")

        elif r["type"] == "generatif":
            score = r["score_llm_juge"]
            lignes.append(f"**Similarité cosinus :** {r['similarite_cosinus']:.3f}\n")
            lignes.append(f"**Score LLM-juge :** {score}/5 — {r['justification_llm_juge']}\n")
            if r.get("notes"):
                lignes.append(f"**Notes :** {r['notes']}\n")
            lignes.append(f"**Réponse de référence :**\n> {r['reponse_reference']}\n")
            lignes.append(f"**Réponse générée :**\n> {r['reponse_generee']}\n")
            lignes.append(f"**Sources utilisées ({len(r['sources'])}) :**\n{_formater_sources(r['sources'])}\n")

        else:
            lignes.append(f"**Nombre de sources :** {r['nombre_sources']}\n")
            lignes.append(f"**Réponse générée :**\n> {r['reponse_generee']}\n")
            lignes.append(f"**Sources utilisées :**\n{_formater_sources(r['sources'])}\n")

        lignes.append("---\n")

    chemin_rapport.write_text("\n".join(lignes), encoding="utf-8")
    return chemin_rapport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=FICHIER_QUESTIONS_DEFAUT,
        help=f"Chemin vers le fichier de questions annotées (défaut : {FICHIER_QUESTIONS_DEFAUT})",
    )
    args = parser.parse_args()

    _verifier_pipeline_disponible()
    questions = charger_questions(args.questions)

    print(f"Évaluation de {len(questions)} questions annotées...\n")
    resultats = evaluer_tout(questions)

    afficher_resume_console(resultats)
    chemin_rapport = sauvegarder_rapport_markdown(resultats)
    print(f"\nRapport détaillé sauvegardé : {chemin_rapport}")


if __name__ == "__main__":
    main()
