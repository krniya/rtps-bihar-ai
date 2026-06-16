"""Initialize the PostgreSQL pgvector extension for the vector-search app."""

from __future__ import annotations

import logging
import os
import sys

import psycopg2
from psycopg2 import OperationalError, Error


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def get_db_settings() -> dict[str, str]:
    """Read database connection settings from environment variables."""

    required_vars = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_PORT"]
    settings = {}

    missing = [name for name in required_vars if not os.getenv(name)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    for name in required_vars:
        settings[name.lower()] = os.getenv(name, "")

    return settings


def enable_vector_extension() -> None:
    """Connect to PostgreSQL and enable pgvector."""

    settings = get_db_settings()

    connection = None
    try:
        connection = psycopg2.connect(
            host=settings["db_host"],
            dbname=settings["db_name"],
            user=settings["db_user"],
            password=settings["db_password"],
            port=settings["db_port"],
        )
        connection.autocommit = True

        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        logging.info("pgvector extension enabled successfully.")

    except OperationalError as exc:
        logging.error("Database connection failed: %s", exc)
        raise
    except Error as exc:
        logging.error("Failed to enable vector extension: %s", exc)
        raise
    finally:
        if connection is not None:
            connection.close()


def main() -> int:
    try:
        enable_vector_extension()
        return 0
    except (ValueError, OperationalError, Error):
        return 1


if __name__ == "__main__":
    sys.exit(main())