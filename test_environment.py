"""
Script de vérification de l'environnement de développement.
Vérifie que toutes les bibliothèques nécessaires au POC RAG
(LangChain, Faiss, Mistral) sont correctement installées et importables.

Usage:
    ./venv/bin/python test_environment.py
"""

import sys


def check_import(label: str, import_fn):
    try:
        import_fn()
        print(f"[OK]    {label}")
        return True
    except ImportError as e:
        print(f"[ECHEC] {label} -> {e}")
        return False


def main():
    print(f"Python : {sys.version}\n")
    results = []

    # --- Coeur LangChain ---
    results.append(check_import("langchain", lambda: __import__("langchain")))
    results.append(check_import("langchain_community", lambda: __import__("langchain_community")))
    results.append(check_import("langchain_mistralai", lambda: __import__("langchain_mistralai")))

    # --- Vector store Faiss ---
    results.append(check_import("faiss (CPU)", lambda: __import__("faiss")))

    # --- Client Mistral ---
    results.append(check_import("mistralai", lambda: __import__("mistralai")))

    # --- Manipulation de données ---
    results.append(check_import("pandas", lambda: __import__("pandas")))

    print()
    if all(results):
        print("Tous les imports ont réussi. L'environnement est prêt.")
        sys.exit(0)
    else:
        print("Certains imports ont échoué. Vérifier l'installation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
