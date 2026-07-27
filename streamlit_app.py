"""
Interface web (Streamlit) pour le chatbot Puls-Events.

Contrairement à chat_cli.py (une boucle `while True` classique dans un
terminal), Streamlit fonctionne sur un modèle d'exécution différent : à
CHAQUE interaction de l'utilisateur (ici, une nouvelle question tapée dans
la zone de saisie), le script entier est ré-exécuté de haut en bas. Il n'y
a pas de boucle explicite à écrire — ce que Streamlit affiche à l'écran
n'est que le résultat de la dernière exécution du script.

Conséquence directe du choix assumé de ne pas garder d'historique de
conversation ici (cohérent avec ask() dans chatbot.py, qui traite déjà
chaque question indépendamment, sans état conservé entre deux appels) :
à chaque nouvelle question posée, l'échange précédent disparaît de l'écran,
puisque rien n'est stocké dans st.session_state entre deux exécutions du
script. Si un jour on veut garder les échanges affichés au fil d'une même
session (juste visuellement, sans changer le comportement du backend), il
suffira d'accumuler les couples (question, résultat) dans
st.session_state et de tous les réafficher à chaque rerun.

Usage :
    streamlit run streamlit_app.py

Dépendance à ajouter si ce n'est pas déjà fait :
    pip install streamlit
"""

import streamlit as st

from src.chatbot import ask

st.set_page_config(
    page_title="Puls-Events — Assistant événements",
    page_icon="🎭",
)

st.title("🎭 Puls-Events")
st.caption("Assistant de recommandation d'événements culturels — Bordeaux Métropole")

question = st.chat_input("Posez votre question (ex : un concert de metal ce mois-ci ?)")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours…"):
            try:
                resultat = ask(question)
            except Exception as e:
                # Volontairement large : dans ce contexte (appel à l'API
                # Mistral), une exception vient presque toujours d'une cause
                # externe (clé API invalide, quota/rate limit dépassé, coupure
                # réseau) et non d'un bug de la logique métier elle-même,
                # déjà couverte par test_chatbot_scenarios.py. Pendant une
                # démo live, un message lisible vaut mieux qu'un traceback
                # brut qui casse la page.
                st.error(f"Une erreur est survenue lors de l'appel à l'API : {e}")
                st.stop()

        st.write(resultat["answer"])

        # resultat["sources"] est une liste de triplets (Document, score,
        # est_hybride) — voir la docstring de ask() dans chatbot.py. Un
        # garde-fou anti-hallucination sans événement pertinent renvoie une
        # liste vide : dans ce cas, pas d'expander à afficher.
        if resultat["sources"]:
            with st.expander(f"Sources ({len(resultat['sources'])} événement(s))"):
                for doc, score, est_hybride in resultat["sources"]:
                    m = doc.metadata
                    marqueur = (
                        " 🔎 *trouvé par filtre exact date/mot-clé*"
                        if est_hybride
                        else ""
                    )
                    st.markdown(
                        f"- **{m.get('title', '?')}** ({m.get('city', '?')}, "
                        f"{(m.get('date_start') or '?')[:10]}) "
                        f"— score={score:.2f}{marqueur}"
                    )

        # None si le garde-fou a bloqué l'appel à Mistral (aucun coût dans
        # ce cas) — voir ask() dans chatbot.py.
        if resultat["usage"]:
            u = resultat["usage"]
            st.caption(
                f"Tokens consommés : {u['input_tokens']} entrée + "
                f"{u['output_tokens']} sortie = {u['total_tokens']} total"
            )
