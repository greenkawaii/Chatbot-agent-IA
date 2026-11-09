"""tools.py — Les outils de l'agent : SQL (partie 1)."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX)\b",
    re.IGNORECASE,
)


class SQLValidationError(ValueError):
    """Requête refusée avant exécution."""


def _mask_sql_literals_and_comments(sql: str) -> tuple[str, list[int]]:
    masked = list(sql)
    semicolons: list[int] = []
    index = 0
    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""

        if char in ("'", '"', "`"):
            quote = char
            masked[index] = " "
            index += 1
            while index < len(sql):
                masked[index] = " " if sql[index] not in "\r\n" else sql[index]
                if sql[index] == quote:
                    if index + 1 < len(sql) and sql[index + 1] == quote:
                        masked[index + 1] = " "
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            continue

        if char == "[":
            masked[index] = " "
            index += 1
            while index < len(sql):
                masked[index] = " " if sql[index] not in "\r\n" else sql[index]
                if sql[index] == "]":
                    index += 1
                    break
                index += 1
            continue

        if char == "-" and next_char == "-":
            while index < len(sql) and sql[index] not in "\r\n":
                masked[index] = " "
                index += 1
            continue

        if char == "/" and next_char == "*":
            masked[index] = masked[index + 1] = " "
            index += 2
            while index < len(sql):
                if sql[index] == "*" and index + 1 < len(sql) and sql[index + 1] == "/":
                    masked[index] = masked[index + 1] = " "
                    index += 2
                    break
                masked[index] = " " if sql[index] not in "\r\n" else sql[index]
                index += 1
            continue

        if char == ";":
            semicolons.append(index)
        index += 1

    return "".join(masked), semicolons


def validate_sql(sql: str) -> str:
    if not isinstance(sql, str):
        raise SQLValidationError("La requête doit être du texte.")

    cleaned = sql.strip()
    fenced = re.fullmatch(r"```(?:sql)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()

    if not cleaned:
        raise SQLValidationError("La requête est vide.")

    masked, semicolons = _mask_sql_literals_and_comments(cleaned)
    if semicolons:
        final_semicolon = semicolons[-1]
        trailing = masked[final_semicolon + 1:]
        if trailing.strip():
            raise SQLValidationError("Une seule instruction SQL est autorisée.")
        if len(semicolons) > 1:
            raise SQLValidationError("Une seule instruction SQL est autorisée.")
        cleaned = cleaned[:final_semicolon].rstrip()
        masked = masked[:final_semicolon].rstrip()

    if not masked.strip():
        raise SQLValidationError("La requête est vide.")
    if not re.match(r"^(SELECT|WITH)\b", masked.lstrip(), flags=re.IGNORECASE):
        raise SQLValidationError("Seules les requêtes SELECT ou WITH sont autorisées.")
    if FORBIDDEN.search(masked):
        raise SQLValidationError("La requête contient un mot-clé interdit.")

    return cleaned


def _format_cell(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value).replace("|", r"\|").replace("\r", " ").replace("\n", " ")


def format_results(
    columns: list[str],
    rows: list[tuple],
    truncated: bool = False,
) -> str:
    count = len(rows)
    if not rows:
        result = "0 ligne(s) : aucun résultat."
    else:
        header = "| " + " | ".join(_format_cell(column) for column in columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        body = [
            "| " + " | ".join(_format_cell(value) for value in row) + " |"
            for row in rows
        ]
        result = f"{count} ligne(s) :\n" + "\n".join([header, separator, *body])
    if truncated:
        result += "\nRésultat tronqué à la limite de lignes."
    return result


class SQLTool:
    name = "query_database"
    description = "Exécute UNE requête SQL SELECT (SQLite) sur la base e-commerce."

    def __init__(self, db_path, max_rows: int = 50) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.is_file():
            raise FileNotFoundError(
                f"Base de données introuvable : {self.db_path}. "
                "Lancez data/create_db.py pour la créer."
            )
        if max_rows < 1:
            raise ValueError("max_rows doit être supérieur ou égal à 1.")
        self.max_rows = max_rows

    def _connect(self) -> sqlite3.Connection:
        uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        try:
            connection.execute("PRAGMA query_only = ON")
        except sqlite3.Error:
            connection.close()
            raise
        return connection

    def run(self, sql: str, params=()) -> str:
        try:
            query = validate_sql(sql)
        except SQLValidationError as exc:
            return f"ERREUR : requête refusée — {exc}"

        try:
            with self._connect() as connection:
                cursor = connection.execute(query, params)
                columns = [description[0] for description in cursor.description or []]
                fetched_rows = cursor.fetchmany(self.max_rows + 1)
        except sqlite3.Error as exc:
            return f"ERREUR SQL : {exc}. Corrigez la requête et réessayez."

        truncated = len(fetched_rows) > self.max_rows
        rows = fetched_rows[: self.max_rows]
        return format_results(columns, rows, truncated=truncated)

    def get_schema(self) -> str:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'table' AND sql IS NOT NULL ORDER BY name"
            ).fetchall()
        return "\n\n".join(row[0] for row in rows)


@dataclass
class Chunk:
    text: str
    source: str
    section: str


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size doit être strictement positif")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap doit être positif et strictement inférieur à chunk_size")

    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    index = 0

    while index < len(words):
        word = words[index]
        added_length = len(word) + (1 if current else 0)
        if not current or current_length + added_length <= chunk_size:
            current.append(word)
            current_length += added_length
            index += 1
            continue

        current_text = " ".join(current)
        chunks.append(current_text)
        if overlap:
            overlap_text = current_text[-overlap:]
            _, separator, remainder = overlap_text.partition(" ")
            current = remainder.split() if separator else []
        else:
            current = []
        current_length = len(" ".join(current))

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_markdown(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    title_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else source
    parts = re.split(r"^##\s+", text, flags=re.MULTILINE)
    chunks: list[Chunk] = []

    for index, part in enumerate(parts):
        heading, separator, content = part.partition("\n")
        if index == 0:
            section = title
            if title_match:
                content = re.sub(r"^#\s+.+(?:\r?\n|$)", "", part, count=1).strip()
            else:
                content = part.strip()
        else:
            section = heading.strip()
            content = content.strip() if separator else ""

        if not content:
            continue
        for piece in chunk_text(content, chunk_size, overlap):
            chunks.append(
                Chunk(
                    text=f"[{title} — {section}] {piece}",
                    source=source,
                    section=section,
                )
            )
    return chunks


def load_corpus(
    docs_dir,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    directory = Path(docs_dir)
    files = sorted(directory.glob("*.md")) if directory.is_dir() else []
    if not files:
        raise FileNotFoundError(f"Aucun fichier Markdown trouvé dans {directory}")

    chunks: list[Chunk] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        chunks.extend(
            chunk_markdown(
                text,
                source=path.name,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
    return chunks


class DocSearchTool:
    name = "search_docs"
    description = (
        "Recherche dans la documentation métier "
        "(KPI, retours, segmentation, données, livraison, reporting)."
    )

    def __init__(self, chunks: list[Chunk], embedder, top_k: int = 3) -> None:
        import faiss

        if not chunks:
            raise ValueError("Il faut au moins un morceau de documentation.")
        if top_k < 1:
            raise ValueError("top_k doit être supérieur ou égal à 1.")

        self.chunks = chunks
        self.embedder = embedder
        self.top_k = top_k
        vectors = self._embed([chunk.text for chunk in chunks])
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("L'embedder doit renvoyer un vecteur par morceau.")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype="float32")

    def search(self, query: str, k: int | None = None) -> list[tuple[float, Chunk]]:
        requested = self.top_k if k is None else k
        if requested <= 0:
            return []
        count = min(requested, len(self.chunks))
        query_vector = self._embed([query])
        if query_vector.ndim != 2 or query_vector.shape[0] != 1:
            raise ValueError("L'embedder doit renvoyer un vecteur pour la question.")
        scores, identifiers = self.index.search(query_vector, count)
        return [
            (float(score), self.chunks[int(identifier)])
            for score, identifier in zip(scores[0], identifiers[0])
            if identifier != -1
        ]

    def run(self, query: str) -> str:
        results = self.search(query)
        if not results:
            return "Aucun passage pertinent trouvé dans la documentation."
        return "\n\n".join(
            f"[Passage {rank} — source : {chunk.source}, "
            f"section « {chunk.section} », score {score:.2f}]\n{chunk.text}"
            for rank, (score, chunk) in enumerate(results, start=1)
        )


def load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)
