"""
Process crawler output from documents/pages/ and convert to ingestion format.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import asdict


def load_crawler_output(pages_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all crawler output JSON files from documents/pages/.

    Args:
        pages_dir: Path to documents/pages directory.

    Returns:
        List of page data dicts.
    """
    pages = []
    for json_file in pages_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                page_data = json.load(f)
                pages.append(page_data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    return pages


def extract_service_metadata(page_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract service metadata from page title and text.

    Args:
        page_data: Page data from crawler.

    Returns:
        Dictionary with department and service_type.
    """
    title = page_data.get("title", "")
    visible_text = page_data.get("visible_text", "")[:500]  # First 500 chars

    # Try to infer department from title or content
    if "Labour" in visible_text or "लेबर" in visible_text:
        department = "Labour Resources Department"
    elif "GAD" in visible_text or "General Administration" in visible_text:
        department = "General Administration Department"
    elif "Plan" in visible_text:
        department = "Plan and Development Department"
    else:
        department = "Other Department"

    # Try to infer service type
    if "certificate" in visible_text.lower() or "प्रमाण" in visible_text:
        service_type = "Certificate Services"
    elif "licence" in visible_text.lower() or "लाइसेंस" in visible_text:
        service_type = "Licensing Services"
    elif "registration" in visible_text.lower() or "निबंधन" in visible_text:
        service_type = "Registration Services"
    else:
        service_type = "Government Services"

    return {
        "department": department,
        "service_type": service_type,
    }


def convert_crawler_to_raw_records(pages_dir: Path, output_file: Path) -> int:
    """
    Convert crawler output to raw records format for ingestion.

    Args:
        pages_dir: Path to documents/pages directory.
        output_file: Path to output JSON file.

    Returns:
        Number of records created.
    """
    pages = load_crawler_output(pages_dir)
    raw_records = []

    for page in pages:
        url = page.get("url", "")
        chunks = page.get("chunks", [])
        metadata = extract_service_metadata(page)

        # Create a raw record for each chunk
        for chunk_text in chunks:
            if chunk_text.strip():  # Only non-empty chunks
                record = {
                    "text": chunk_text,
                    "source_url": url,
                    "department": metadata["department"],
                    "service_type": metadata["service_type"],
                }
                raw_records.append(record)

    # Write to output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(raw_records, f, ensure_ascii=False, indent=2)

    print(f"✓ Converted {len(pages)} pages to {len(raw_records)} records")
    print(f"✓ Output saved to {output_file}")

    return len(raw_records)


def main():
    """Main entry point."""
    project_root = Path(__file__).resolve().parent.parent

    pages_dir = project_root / "documents" / "pages"
    output_file = project_root / "ingestion" / "raw_records" / "raw_records_from_crawler.json"

    if not pages_dir.exists():
        print(f"❌ Error: {pages_dir} not found")
        return

    print("Converting crawler output to raw records format...")
    count = convert_crawler_to_raw_records(pages_dir, output_file)

    if count > 0:
        print(f"\n✓ Ready for ingestion!")
        print(f"Run: uv run .\\ingestion\\vector_ingest.py --input-json {output_file}")
    else:
        print("❌ No records created")


if __name__ == "__main__":
    main()
