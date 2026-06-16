"""Chunk raw text, create Gemini embeddings, and store them in PostgreSQL.

This module is intentionally separate from the db/ bootstrap code so the
database setup and the vector ingest pipeline stay cleanly isolated.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Integer, String, Text, create_engine, func, insert, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMENSION = 3072


@dataclass(slots=True)
class RawTextRecord:
    """One source document and the metadata that should travel with it."""

    text: str
    source_url: str
    department: str
    service_type: str


class Base(DeclarativeBase):
    pass


class TextChunk(Base):
    """Explicit table for searchable text chunks and their embeddings."""

    __tablename__ = "text_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_url: Mapped[str] = mapped_column(String(1024), nullable=False, index=True)
    department: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    service_type: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSION), nullable=False)
    extra_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


def get_database_url() -> str:
    """Build the SQLAlchemy connection URL from environment variables."""

    required_vars = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_PORT"]
    missing = [name for name in required_vars if not os.getenv(name)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    return (
        "postgresql+psycopg2://"
        f"{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}@"
        f"{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
    )


def ensure_vector_extension(engine) -> None:
    """Make sure pgvector is enabled before table creation or inserts."""

    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))


def build_text_splitter() -> RecursiveCharacterTextSplitter:
    """Use the requested chunk size and overlap settings."""

    return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def build_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create the LangChain embedding client for Gemini text embeddings."""

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set")

    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=gemini_api_key)


def load_raw_records(input_path: Path) -> list[RawTextRecord]:
    """Load raw records from a JSON file.

    The file must contain a list of objects with: text, source_url, department,
    and service_type.
    """

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Input JSON must contain a list of records")

    records: list[RawTextRecord] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Record #{index} is not an object")

        try:
            records.append(
                RawTextRecord(
                    text=str(item["text"]),
                    source_url=str(item["source_url"]),
                    department=str(item["department"]),
                    service_type=str(item["service_type"]),
                )
            )
        except KeyError as exc:
            raise ValueError(f"Record #{index} is missing field: {exc.args[0]}") from exc

    return records


def chunk_records(records: Iterable[RawTextRecord]) -> list[dict]:
    """Split each raw text record into fixed-size LangChain chunks."""

    splitter = build_text_splitter()
    chunk_rows: list[dict] = []

    for record in records:
        chunks = splitter.split_text(record.text)
        for chunk_index, chunk_text in enumerate(chunks):
            chunk_rows.append(
                {
                    "source_url": record.source_url,
                    "department": record.department,
                    "service_type": record.service_type,
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "extra_metadata": {
                        "source_url": record.source_url,
                        "department": record.department,
                        "service_type": record.service_type,
                    },
                }
            )

    return chunk_rows


def embed_chunk_texts(chunk_rows: list[dict], embeddings: GoogleGenerativeAIEmbeddings) -> list[list[float]]:
    """Batch-embed the chunk texts for efficient database writes."""

    texts = [row["chunk_text"] for row in chunk_rows]
    if not texts:
        return []

    return embeddings.embed_documents(texts)


def create_schema(engine) -> None:
    """Create the table structure if it does not already exist."""

    Base.metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE IF EXISTS text_chunks ALTER COLUMN embedding TYPE vector(3072) USING embedding::vector(3072);"))


def persist_chunks(session: Session, chunk_rows: list[dict], vectors: list[list[float]]) -> int:
    """Bulk insert chunk rows with their corresponding embeddings."""

    if len(chunk_rows) != len(vectors):
        raise ValueError("Chunk rows and vectors must have the same length")

    rows = []
    for chunk_row, vector in zip(chunk_rows, vectors, strict=True):
        rows.append(
            {
                "source_url": chunk_row["source_url"],
                "department": chunk_row["department"],
                "service_type": chunk_row["service_type"],
                "chunk_index": chunk_row["chunk_index"],
                "chunk_text": chunk_row["chunk_text"],
                "embedding": vector,
                "extra_metadata": chunk_row["extra_metadata"],
            }
        )

    if rows:
        session.execute(insert(TextChunk), rows)
        session.commit()

    return len(rows)


def ingest_raw_text(records: list[RawTextRecord]) -> int:
    """Run the full chunk -> embed -> persist pipeline."""

    engine = create_engine(get_database_url())
    ensure_vector_extension(engine)
    create_schema(engine)

    embeddings = build_embeddings()
    chunk_rows = chunk_records(records)
    vectors = embed_chunk_texts(chunk_rows, embeddings)

    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        saved_count = persist_chunks(session, chunk_rows, vectors)

    return saved_count


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest raw text into PostgreSQL with pgvector")
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="JSON file containing a list of raw text records",
    )
    return parser


def main() -> int:
    try:
        args = build_argument_parser().parse_args()
        records = load_raw_records(args.input_json)
        saved_count = ingest_raw_text(records)
        logging.info("Saved %s text chunks into PostgreSQL.", saved_count)
        return 0
    except Exception as exc:
        logging.error("Vector ingest failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())