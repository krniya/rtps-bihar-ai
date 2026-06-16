# RTPS Bihar AI

## Layout

- `crawler/` contains the Playwright-based portal crawler.
- `db/` contains PostgreSQL bootstrap scripts, including `init_db.py`.
- `ingestion/` contains the Gemini + pgvector text-ingest pipeline.
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

Run the web crawler:

```bash
uv run .\crawler\rtps_crawler.py
```