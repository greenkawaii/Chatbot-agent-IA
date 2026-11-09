"""Évalue le hit rate de la recherche documentaire à plusieurs tailles de chunks."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from tools import DocSearchTool, load_corpus, load_embedder


JEU_EVAL = [
    ("Combien de jours pour retourner un article ?", "02_politique_retours.md"),
    ("Comment est défini le chiffre d'affaires ?", "01_glossaire_kpi.md"),
    ("Quels sont les critères pour devenir VIP ?", "03_segmentation_clients.md"),
    (
        "Quelles colonnes contient la table lignes_commande ?",
        "04_dictionnaire_donnees.md",
    ),
    (
        "À partir de quel montant la livraison est-elle gratuite ?",
        "05_livraison.md",
    ),
    (
        "Peut-on afficher l'email d'un client dans un rapport ?",
        "06_regles_reporting.md",
    ),
    ("Qu'est-ce qu'un client actif ?", "01_glossaire_kpi.md"),
    (
        "Les produits de beauté peuvent-ils être retournés ?",
        "02_politique_retours.md",
    ),
]


def hit_rate(tool: DocSearchTool, k: int) -> float:
    hits = sum(
        any(chunk.source == expected_source for _, chunk in tool.search(question, k=k))
        for question, expected_source in JEU_EVAL
    )
    return hits / len(JEU_EVAL)


def main() -> None:
    embedder = load_embedder(config.EMBEDDING_MODEL)
    for chunk_size in (300, 500, 800):
        chunks = load_corpus(
            config.DOCS_DIR,
            chunk_size=chunk_size,
            overlap=min(config.CHUNK_OVERLAP, chunk_size - 1),
        )
        tool = DocSearchTool(chunks, embedder, top_k=3)
        print(
            f"chunk_size={chunk_size}: "
            f"hit rate @1 = {hit_rate(tool, 1):.0%}, "
            f"hit rate @3 = {hit_rate(tool, 3):.0%}"
        )


if __name__ == "__main__":
    main()
