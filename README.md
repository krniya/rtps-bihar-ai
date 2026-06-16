# RTPS Bihar AI

## Layout

- `crawler/` contains the Playwright-based portal crawler.
- `db/` contains PostgreSQL bootstrap scripts, including `init_db.py`.
- `ingestion/` contains the Gemini + pgvector text-ingest pipeline.
- `retrieval/` contains semantic search and document retrieval functions using PGVector.
- `documents/` stores crawler output, downloaded PDFs, and metadata.

## Setup

Install all dependencies from the root:

```bash
pip install -r requirements.txt
```

Or with `uv`:

```bash
uv pip install -r requirements.txt
```

## Vector ingest

Run the vector pipeline:

```bash
uv run .\ingestion\vector_ingest.py --input-json .\ingestion\sample_raw_records.json
```

## Database bootstrap

Enable the pgvector extension:

```bash
uv run .\db\init_db.py
```

## Crawler

Run the web crawler to scrape the Bihar RTPS portal:

```bash
uv run .\crawler\rtps_crawler.py
```

Crawler outputs are saved to:
- `documents/pages/` - Page content and chunks (JSON)
- `documents/pdfs/` - Downloaded PDF files
- `documents/metadata/crawl_index.json` - Crawl index

## Processing crawler output

Convert crawler output to raw records format for vector ingestion:

```bash
uv run .\ingestion\process_crawler_output.py
```

This generates `ingestion/raw_records/raw_records_from_crawler.json` from the documents in `documents/pages/`.

## Processing PDFs

Extract text from PDF documents (39 user manuals, guides, forms):

```bash
uv run .\ingestion\process_all_documents.py
```

This extracts text from all PDFs in `documents/pdfs/` and combines with crawler output to create `ingestion/raw_records/raw_records_from_all_documents.json` (~315 chunks).

## Vector ingestion

Ingest raw records into PostgreSQL with Gemini embeddings:

**Option 1: Crawler output only (219 chunks)**
```bash
uv run .\ingestion\vector_ingest.py --input-json .\ingestion\raw_records\raw_records_from_crawler.json
```

**Option 2: Crawler + PDFs combined (315 chunks)**
```bash
uv run .\ingestion\vector_ingest.py --input-json .\ingestion\raw_records\raw_records_from_all_documents.json
```

**Option 3: Sample data**
```bash
uv run .\ingestion\vector_ingest.py --input-json .\ingestion\sample_raw_records.json
```

## Document Retrieval

Query the vector database using semantic search.

**Workflow:**
1. Crawl the portal → `documents/pages/`
2. Process crawler output → `ingestion/raw_records_from_crawler.json`
3. Ingest to PostgreSQL → pgvector embeddings
4. Query using semantic search → relevant documents

### Simple semantic search

```bash
uv run .\retrieval\example_retrieval.py
```

### In your Python code

```python
from retrieval.retrieve_docs import retrieve_relevant_docs, retrieve_with_sql_filter

# Basic semantic search
results = retrieve_relevant_docs(
    query="How to apply for residential certificate?",
    top_k=4
)

# Filtered by department
results = retrieve_relevant_docs(
    query="certificate services",
    department_filter="General Administration Department",
    top_k=4
)

# Advanced filtering (department + service type)
results = retrieve_with_sql_filter(
    query="licence registration",
    department="Labour Resources Department",
    service_type="Other Services",
    top_k=4
)

# Each result contains:
# - chunk_text: The document chunk
# - source_url: URL where chunk came from
# - department: Department name
# - service_type: Service type
# - similarity_score: Relevance score (for SQL-filtered search)
```