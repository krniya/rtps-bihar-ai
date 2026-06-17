"""
Semantic document retrieval using LangChain + PGVector.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime


# Load environment variables
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Database configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rtps_vectors")
DB_USER = os.getenv("DB_USER", "rtps_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "change_me")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Connection string for LangChain PGVector
CONNECTION_STRING = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Embedding configuration
EMBEDDING_DIMENSION = 1536


def get_pgvector_store() -> PGVector:
    """Return a LangChain PGVector store connected to the text_chunks table."""

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY,
    )

    return PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
        collection_name="text_chunks",
        use_jsonb=False,
    )


def get_pgvector_retriever(
    department_filter: Optional[str] = None,
    top_k: int = 4,
):
    """Return a LangChain retriever backed by the pgvector store."""

    vector_store = get_pgvector_store()

    if department_filter:
        return vector_store.as_retriever(
            search_kwargs={
                "k": top_k,
                "filter": {"department": department_filter},
            }
        )

    return vector_store.as_retriever(search_kwargs={"k": top_k})


def retrieve_relevant_docs(
    query: str,
    department_filter: Optional[str] = None,
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant document chunks using semantic similarity search.

    Args:
        query: Natural language query string.
        department_filter: Optional department name to filter results.
        top_k: Number of top results to return (default: 4).

    Returns:
        List of dictionaries containing chunk text and metadata.
        Format: [
            {
                "chunk_id": str,
                "chunk_text": str,
                "source_url": str,
                "department": str,
                "service_type": str,
                "chunk_index": int,
                "similarity_score": float,
                "created_at": str,
            },
            ...
        ]
    """

    try:
        retriever = get_pgvector_retriever(
            department_filter=department_filter,
            top_k=top_k,
        )

        # Perform semantic search
        results = retriever.invoke(query)

        # Format results with metadata
        formatted_results = []
        for i, doc in enumerate(results):
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            formatted_results.append(
                {
                    "chunk_id": metadata.get("id", "unknown"),
                    "chunk_text": doc.page_content,
                    "source_url": metadata.get("source_url", ""),
                    "department": metadata.get("department", ""),
                    "service_type": metadata.get("service_type", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "similarity_score": metadata.get("similarity_score", None),
                    "created_at": str(metadata.get("created_at", "")),
                }
            )

        return formatted_results

    except Exception as e:
        print(f"Error during document retrieval: {e}")
        raise


def retrieve_with_sql_filter(
    query: str,
    department: Optional[str] = None,
    service_type: Optional[str] = None,
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """
    Advanced retrieval with direct SQL filtering for hybrid search.

    Args:
        query: Natural language query string.
        department: Optional department filter.
        service_type: Optional service type filter.
        top_k: Number of results to return.

    Returns:
        List of relevant document chunks with metadata.
    """

    try:
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY,
        )

        # Generate embedding for query
        query_embedding = embeddings.embed_query(query)

        # Direct SQL connection for filtered vector search
        engine = create_engine(CONNECTION_STRING)
        with Session(engine) as session:
            # Build base query
            sql_query = """
                SELECT id, source_url, department, service_type, chunk_index,
                       chunk_text, created_at,
                       1 - (embedding <=> %s::vector) as similarity_score
                FROM text_chunks
                WHERE 1=1
            """
            params = [str(query_embedding)]

            # Apply department filter
            if department:
                sql_query += " AND department = %s"
                params.append(department)

            # Apply service type filter
            if service_type:
                sql_query += " AND service_type = %s"
                params.append(service_type)

            # Order by similarity and limit
            sql_query += " ORDER BY similarity_score DESC LIMIT %s"
            params.append(top_k)

            # Execute query
            result = session.execute(sql_query, params)
            rows = result.fetchall()

            # Format results
            formatted_results = []
            for row in rows:
                formatted_results.append(
                    {
                        "chunk_id": str(row[0]),
                        "source_url": row[1],
                        "department": row[2],
                        "service_type": row[3],
                        "chunk_index": row[4],
                        "chunk_text": row[5],
                        "created_at": str(row[6]),
                        "similarity_score": float(row[7]) if row[7] else 0.0,
                    }
                )

            return formatted_results

    except Exception as e:
        print(f"Error during SQL-filtered retrieval: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Document Retrieval Example")
    print("=" * 60)

    # Test query
    test_query = "How to apply for residential certificate?"

    # Test 1: Simple semantic search
    print("\n1. Simple Semantic Search:")
    print(f"Query: {test_query}\n")
    try:
        results = retrieve_relevant_docs(query=test_query, top_k=4)
        for i, doc in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Department: {doc['department']}")
            print(f"  Service Type: {doc['service_type']}")
            print(f"  Text: {doc['chunk_text'][:100]}...")
            print(f"  URL: {doc['source_url']}\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # Test 2: Filtered search by department
    print("2. Filtered by Department (General Administration):")
    try:
        results = retrieve_relevant_docs(
            query=test_query,
            department_filter="General Administration Department",
            top_k=2,
        )
        for i, doc in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Department: {doc['department']}")
            print(f"  Service Type: {doc['service_type']}")
            print(f"  Text: {doc['chunk_text'][:100]}...\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # Test 3: Advanced SQL filtering
    print("3. Advanced SQL Filtering (Department + Service Type):")
    try:
        results = retrieve_with_sql_filter(
            query=test_query,
            department="General Administration Department",
            service_type="RTPS Services",
            top_k=2,
        )
        for i, doc in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Department: {doc['department']}")
            print(f"  Service Type: {doc['service_type']}")
            print(f"  Similarity: {doc['similarity_score']:.4f}")
            print(f"  Text: {doc['chunk_text'][:100]}...\n")
    except Exception as e:
        print(f"Error: {e}\n")
