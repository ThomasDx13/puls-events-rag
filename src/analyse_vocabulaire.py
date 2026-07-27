"""
Script de diagnostic (pas utilisé par le chatbot en production) : affiche
les mots-clés actuellement retenus dans le vocabulaire "rare" utilisé par
la recherche hybride de chatbot.py (_construire_vocabulaire_mots_cles()),
triés du plus fréquent au moins fréquent.

Objectif : repérer visuellement les mots-clés qui, bien que sous le seuil
RAG_KEYWORD_MAX_FREQUENCY, sont en réalité des connecteurs de catégorie peu
discriminants (ex: "cours", partagé par des dizaines d'activités sans
rapport entre elles : danse, dessin, théâtre...) plutôt que de vrais
descripteurs de contenu (ex: "bachata", "vinyle"...). Intuition statistique
à vérifier visuellement : les connecteurs génériques ont plus de chances
d'être proches du haut de la liste (fréquents, mais pas assez pour
franchir le seuil) ; les vrais descripteurs spécifiques sont souvent très
rares (1-2 occurrences), donc plutôt en bas.

Réutilise _normaliser() importé depuis src.chatbot : même normalisation
(casse + accents) que celle réellement utilisée pour construire le
vocabulaire en production — pas de logique dupliquée, pas de risque de
divergence entre ce diagnostic et le comportement réel du chatbot.

Note : comme _construire_vocabulaire_mots_cles(), ce script compte les
mots-clés par CHUNK et non par événement unique (un événement à plusieurs
chunks voit ses mots-clés comptés plusieurs fois) — c'est un miroir fidèle
du comportement réel, pas une version "corrigée".

Usage :
    python -m src.analyse_vocabulaire
    # Ne montrer que les mots-clés apparaissant au moins 5 fois (exclut la
    # longue traîne des descripteurs rares/légitimes, qui ne peuvent pas
    # être des connecteurs génériques puisqu'ils ne sont justement pas
    # partagés par beaucoup d'événements) :
    python -m src.analyse_vocabulaire --min-count 5
    # Écrire la sortie complète dans un fichier plutôt que le terminal :
    python -m src.analyse_vocabulaire --output vocabulaire.txt
"""

import argparse
import json

from src import config
from src.chatbot import _normaliser


def compter_mots_cles() -> tuple[dict[str, int], int]:
    """Même logique de comptage que _construire_vocabulaire_mots_cles()
    dans chatbot.py, mais on garde le compteur complet au lieu de ne
    retourner que le set filtré — c'est justement ce compteur qu'on veut
    inspecter ici."""
    if not config.CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"{config.CHUNKS_FILE} introuvable. Lance d'abord `python -m src.vectorize`."
        )
    chunks = json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))

    compteur: dict[str, int] = {}
    for chunk in chunks:
        for mot_cle in chunk["metadata"].get("keywords") or []:
            mot_cle_normalise = _normaliser(mot_cle)
            compteur[mot_cle_normalise] = compteur.get(mot_cle_normalise, 0) + 1

    return compteur, len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-count", type=int, default=1,
        help="N'afficher que les mots-clés apparaissant au moins N fois "
             "(par défaut : 1, donc tout le vocabulaire hybride). Filtre par "
             "fréquence plutôt que par rang : plus pertinent ici, puisque la "
             "zone à risque (connecteurs génériques) est celle des fréquences "
             "élevées-mais-sous-le-seuil, pas un nombre fixe de lignes.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Chemin d'un fichier où écrire la sortie (en plus de l'affichage "
             "terminal). Utile pour grep/rechercher un mot précis sans "
             "dépendre de la hauteur du terminal.",
    )
    args = parser.parse_args()

    compteur, n_chunks = compter_mots_cles()
    seuil_absolu = n_chunks * config.RAG_KEYWORD_MAX_FREQUENCY

    # Seuls les mots-clés SOUS le seuil sont réellement utilisés pour le
    # matching hybride aujourd'hui (les autres sont déjà exclus par
    # _construire_vocabulaire_mots_cles()) — pas la peine d'afficher le
    # reste, ça ne changerait rien au comportement du chatbot.
    sous_seuil = [
        (mot, n) for mot, n in compteur.items()
        if n <= seuil_absolu and n >= args.min_count
    ]
    sous_seuil.sort(key=lambda item: item[1], reverse=True)

    lignes = [
        f"{n_chunks} chunks — seuil RAG_KEYWORD_MAX_FREQUENCY = "
        f"{config.RAG_KEYWORD_MAX_FREQUENCY} ({seuil_absolu:.1f} occurrences max)",
        "",
        f"{len(sous_seuil)} mots-clés affichés (sur "
        f"{sum(1 for n in compteur.values() if n <= seuil_absolu)} au total dans "
        f"le vocabulaire hybride), triés du plus fréquent au moins fréquent :",
        "",
    ]
    lignes += [f"  {n:>4}  {mot}" for mot, n in sous_seuil]

    texte = "\n".join(lignes)
    print(texte)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(texte + "\n")
        print(f"\n[Sortie également écrite dans {args.output}]")


if __name__ == "__main__":
    main()
