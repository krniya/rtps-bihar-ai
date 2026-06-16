"""
Extract text from PDF documents and convert to ingestion format.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF file.

    Args:
        pdf_path: Path to PDF file.

    Returns:
        Extracted text from PDF.
    """
    if pdfplumber is None:
        print(f"Warning: pdfplumber not installed. Skipping {pdf_path.name}")
        return ""

    text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
    except Exception as e:
        print(f"Error extracting text from {pdf_path.name}: {e}")
        return ""

    return "\n\n".join(text)


def infer_pdf_metadata(filename: str, text: str) -> Dict[str, str]:
    """
    Infer department and service type from PDF filename and content.

    Args:
        filename: PDF filename.
        text: Extracted text content.

    Returns:
        Dictionary with department and service_type.
    """
    text_lower = text.lower()[:1000]  # First 1000 chars

    # Infer department from filename
    if "LRD" in filename or "labour" in text_lower:
        department = "Labour Resources Department"
    elif "GAD" in filename or "general" in text_lower:
        department = "General Administration Department"
    elif "hotel" in filename.lower() or "tourism" in text_lower:
        department = "Tourism Department"
    else:
        department = "General Administration Department"

    # Infer service type from filename
    if "manual" in filename.lower() or "user" in filename.lower():
        service_type = "User Manual"
    elif "form" in filename.lower():
        service_type = "Form"
    elif "service" in filename.lower():
        service_type = "Service Guide"
    elif "brochure" in filename.lower():
        service_type = "Brochure"
    else:
        service_type = "Documentation"

    return {
        "department": department,
        "service_type": service_type,
    }


def split_text_into_chunks(text: str, chunk_size: int = 2000) -> List[str]:
    """
    Split text into chunks for ingestion.

    Args:
        text: Full text to split.
        chunk_size: Approximate size of each chunk.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    chunks = []
    lines = text.split("\n")
    current_chunk = []
    current_size = 0

    for line in lines:
        line_size = len(line) + 1  # +1 for newline
        if current_size + line_size > chunk_size and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_size = line_size
        else:
            current_chunk.append(line)
            current_size += line_size

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return [c for c in chunks if c.strip()]


def convert_pdfs_to_records(pdfs_dir: Path) -> List[Dict[str, Any]]:
    """
    Convert PDF documents to raw records for ingestion.

    Args:
        pdfs_dir: Path to PDFs directory.

    Returns:
        List of raw records.
    """
    records = []

    pdf_files = list(pdfs_dir.glob("*.pdf"))
    print(f"Processing {len(pdf_files)} PDF files...")

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"  [{i}/{len(pdf_files)}] {pdf_path.name}...", end=" ", flush=True)

        text = extract_text_from_pdf(pdf_path)
        if not text:
            print("(no text extracted)")
            continue

        metadata = infer_pdf_metadata(pdf_path.name, text)
        chunks = split_text_into_chunks(text)

        for chunk_text in chunks:
            if chunk_text.strip():
                record = {
                    "text": chunk_text,
                    "source_url": f"file:///{pdf_path.name}",  # Local file reference
                    "department": metadata["department"],
                    "service_type": metadata["service_type"],
                }
                records.append(record)

        print(f"({len(chunks)} chunks)")

    return records


def merge_records(crawler_records: List[Dict], pdf_records: List[Dict]) -> List[Dict]:
    """
    Merge crawler and PDF records.

    Args:
        crawler_records: Records from crawler.
        pdf_records: Records from PDFs.

    Returns:
        Combined list of records.
    """
    return crawler_records + pdf_records


def process_all_documents(
    project_root: Path, output_file: Path, include_pdfs: bool = True
) -> int:
    """
    Process all documents (crawler + PDFs) and create raw records.

    Args:
        project_root: Root of project.
        output_file: Output JSON file path.
        include_pdfs: Whether to include PDF processing.

    Returns:
        Number of records created.
    """
    # Load crawler records
    pages_dir = project_root / "documents" / "pages"
    crawler_records = []

    print("Loading crawler output...")
    if pages_dir.exists():
        for json_file in pages_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    page_data = json.load(f)
                    url = page_data.get("url", "")
                    chunks = page_data.get("chunks", [])

                    # Infer metadata
                    text = page_data.get("visible_text", "")[:500]
                    if "labour" in text.lower() or "labour" in text.lower():
                        department = "Labour Resources Department"
                    else:
                        department = "General Administration Department"

                    if "certificate" in text.lower():
                        service_type = "Certificate Services"
                    elif "licence" in text.lower():
                        service_type = "Licensing Services"
                    else:
                        service_type = "Government Services"

                    for chunk in chunks:
                        if chunk.strip():
                            record = {
                                "text": chunk,
                                "source_url": url,
                                "department": department,
                                "service_type": service_type,
                            }
                            crawler_records.append(record)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

    print(f"✓ Loaded {len(crawler_records)} chunks from crawler\n")

    # Load PDF records
    pdf_records = []
    if include_pdfs:
        pdfs_dir = project_root / "documents" / "pdfs"
        if pdfs_dir.exists():
            pdf_records = convert_pdfs_to_records(pdfs_dir)
            print(f"✓ Extracted {len(pdf_records)} chunks from PDFs\n")

    # Merge and save
    all_records = merge_records(crawler_records, pdf_records)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"✓ Total records created: {len(all_records)}")
    print(f"✓ Output saved to {output_file}")

    return len(all_records)


def main():
    """Main entry point."""
    project_root = Path(__file__).resolve().parent.parent

    output_file = project_root / "ingestion" / "raw_records" / "raw_records_from_all_documents.json"

    print("Processing crawler output + PDFs...")
    print("=" * 60)

    count = process_all_documents(project_root, output_file, include_pdfs=True)

    if count > 0:
        print("\n" + "=" * 60)
        print("✓ Ready for ingestion!")
        print(f"Run: uv run .\\ingestion\\vector_ingest.py --input-json {output_file}")
    else:
        print("❌ No records created")


if __name__ == "__main__":
    main()
