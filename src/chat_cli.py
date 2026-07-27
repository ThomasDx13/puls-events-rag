"""
Interface en ligne de commande pour interagir avec le chatbot Puls-Events.

Chaque question est traitée indépendamment (pas d'historique de
conversation, conformément à la consigne du POC — voir chatbot.py).

Usage :
    python -m src.chat_cli
"""

from src.chatbot import ask


def main() -> None:
    print("=" * 70)
    print("Chatbot Puls-Events — recommandations d'événements (Bordeaux Métropole)")
    print("Tape 'quit' (ou Ctrl+C) pour quitter.")
    print("=" * 70)

    while True:
        try:
            question = input("\nVous : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÀ bientôt !")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("À bientôt !")
            break
        if not question:
            continue

        resultat = ask(question)

        print(f"\nChatbot : {resultat['answer']}")

        if resultat["sources"]:
            print("\nSources (événements utilisés pour cette réponse) :")
            for doc, score, est_hybride in resultat["sources"]:
                m = doc.metadata
                marqueur = " [trouvé par filtre exact date/mot-clé]" if est_hybride else ""
                print(f"  - {m['title']} ({m['city']}, {(m.get('date_start') or '?')[:10]}) "
                      f"— score={score:.2f}{marqueur}")

        if resultat["usage"]:
            u = resultat["usage"]
            print(f"\n[Tokens consommés : {u['input_tokens']} entrée + "
                  f"{u['output_tokens']} sortie = {u['total_tokens']} total]")


if __name__ == "__main__":
    main()
