"""Async crawler for dynamic government portals.

This script uses Playwright to render JavaScript-heavy pages, BeautifulSoup to
extract visible text and links, and LangChain's RecursiveCharacterTextSplitter
to chunk long guideline text for downstream processing.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)


@dataclass(slots=True)
class CrawlConfig:
    start_url: str
    output_dir: Path
    target_texts: list[str]
    politeness_delay: float = 2.0
    max_pages: int = 20
    max_depth: int = 2
    navigation_timeout_ms: int = 60000


def ensure_directories(output_dir: Path) -> dict[str, Path]:
    """Create the output layout used by the crawler."""

    pages_dir = output_dir / "pages"
    pdfs_dir = output_dir / "pdfs"
    metadata_dir = output_dir / "metadata"

    for directory in (output_dir, pages_dir, pdfs_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "root": output_dir,
        "pages": pages_dir,
        "pdfs": pdfs_dir,
        "metadata": metadata_dir,
    }


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sanitize_filename(value: str, fallback: str) -> str:
    """Create a Windows-safe file name from arbitrary text or URLs."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or fallback


def is_same_domain(base_url: str, candidate_url: str) -> bool:
    base_domain = urlparse(base_url).netloc.lower()
    candidate_domain = urlparse(candidate_url).netloc.lower()
    return bool(candidate_domain) and candidate_domain == base_domain


def make_absolute_url(base_url: str, href: str) -> str:
    return urljoin(base_url, href)


def build_chunk_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def extract_visible_text_and_links(page_html: str, base_url: str) -> tuple[str, list[str], list[str]]:
    """Parse the rendered HTML and return visible text, PDF links, and page links."""

    soup = BeautifulSoup(page_html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "canvas"]):
        tag.decompose()

    visible_text = normalize_whitespace(soup.get_text(separator=" ", strip=True))

    pdf_links: list[str] = []
    internal_links: list[str] = []
    seen_links: set[str] = set()

    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "").strip()
        if not href:
            continue

        absolute_url = make_absolute_url(base_url, href)
        href_lower = absolute_url.lower()
        anchor_text = normalize_whitespace(anchor.get_text(" ", strip=True)).lower()

        if absolute_url in seen_links:
            continue
        seen_links.add(absolute_url)

        if any(
            token in href_lower or token in anchor_text
            for token in (".pdf", "pdf", "document", "resource")
        ):
            pdf_links.append(absolute_url)
            continue

        if href_lower.startswith(("mailto:", "tel:", "javascript:")):
            continue

        internal_links.append(absolute_url)

    return visible_text, pdf_links, internal_links


async def expand_dynamic_accordions(page, target_texts: Iterable[str]) -> None:
    """Click matching text nodes so hidden accordion content becomes visible."""

    for text in target_texts:
        try:
            locator = page.get_by_text(text, exact=False).first
            await locator.scroll_into_view_if_needed(timeout=5000)
            await locator.click(timeout=5000)
            await page.wait_for_load_state("networkidle", timeout=10000)
        except PlaywrightTimeoutError:
            logging.warning("Accordion text not found or not clickable: %s", text)
        except Exception as exc:  # pragma: no cover - defensive logging only
            logging.warning("Failed to expand accordion '%s': %s", text, exc)


async def write_text_file(path: Path, content: str) -> None:
    await asyncio.to_thread(path.write_text, content, encoding="utf-8")


async def write_json_file(path: Path, payload: dict) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    await asyncio.to_thread(path.write_text, serialized, encoding="utf-8")


async def download_pdf(session: aiohttp.ClientSession, pdf_url: str, destination: Path) -> dict[str, str]:
    """Download a PDF and store it locally."""

    try:
        async with session.get(pdf_url) as response:
            response.raise_for_status()
            pdf_bytes = await response.read()
            await asyncio.to_thread(destination.write_bytes, pdf_bytes)
            return {
                "url": pdf_url,
                "saved_to": str(destination),
                "status": "downloaded",
            }
    except Exception as exc:
        logging.warning("Failed to download PDF %s: %s", pdf_url, exc)
        return {
            "url": pdf_url,
            "saved_to": str(destination),
            "status": f"failed: {exc}",
        }


def build_pdf_destination(pdfs_dir: Path, pdf_url: str) -> Path:
    parsed = urlparse(pdf_url)
    candidate_name = Path(parsed.path).name
    if not candidate_name:
        digest = hashlib.sha256(pdf_url.encode("utf-8")).hexdigest()[:12]
        candidate_name = f"document_{digest}.pdf"

    if not candidate_name.lower().endswith(".pdf"):
        candidate_name = f"{candidate_name}.pdf"

    safe_name = sanitize_filename(candidate_name, "document.pdf")
    return pdfs_dir / safe_name


async def save_page_artifacts(
    page_url: str,
    page_title: str,
    visible_text: str,
    pdf_links: list[str],
    pages_dir: Path,
    splitter: RecursiveCharacterTextSplitter,
) -> dict:
    """Persist text chunks and page metadata for a single crawled page."""

    chunks = splitter.split_text(visible_text) if visible_text else []
    page_slug = sanitize_filename(page_title or urlparse(page_url).path or "page", "page")
    page_hash = hashlib.sha256(page_url.encode("utf-8")).hexdigest()[:10]
    record_name = f"{page_slug}_{page_hash}.json"
    record_path = pages_dir / record_name

    payload = {
        "url": page_url,
        "title": page_title,
        "visible_text": visible_text,
        "chunks": chunks,
        "pdf_links": pdf_links,
    }
    await write_json_file(record_path, payload)
    return {"record_path": str(record_path), "chunk_count": len(chunks)}


async def crawl_site(config: CrawlConfig) -> list[dict]:
    """Crawl the configured portal and collect page text plus PDF documents."""

    directories = ensure_directories(config.output_dir)
    splitter = build_chunk_splitter()
    crawl_results: list[dict] = []
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(config.start_url, 0)]

    timeout = aiohttp.ClientTimeout(total=120)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1440, "height": 1200},
            locale="en-US",
        )
        page = await context.new_page()

        async with aiohttp.ClientSession(timeout=timeout) as http_session:
            while queue and len(visited) < config.max_pages:
                current_url, depth = queue.pop(0)
                if current_url in visited:
                    continue

                visited.add(current_url)
                logging.info("Visiting %s (depth=%s)", current_url, depth)

                try:
                    await page.goto(current_url, wait_until="networkidle", timeout=config.navigation_timeout_ms)
                    await page.wait_for_load_state("networkidle", timeout=config.navigation_timeout_ms)
                    await asyncio.sleep(config.politeness_delay)

                    await expand_dynamic_accordions(page, config.target_texts)
                    await page.wait_for_load_state("networkidle", timeout=10000)

                    page_html = await page.content()
                    page_title = normalize_whitespace(await page.title())
                    visible_text, pdf_links, internal_links = extract_visible_text_and_links(page_html, current_url)

                    page_result = await save_page_artifacts(
                        page_url=current_url,
                        page_title=page_title,
                        visible_text=visible_text,
                        pdf_links=pdf_links,
                        pages_dir=directories["pages"],
                        splitter=splitter,
                    )

                    downloaded_pdfs: list[dict[str, str]] = []
                    for pdf_url in pdf_links:
                        pdf_destination = build_pdf_destination(directories["pdfs"], pdf_url)
                        if pdf_destination.exists():
                            downloaded_pdfs.append(
                                {
                                    "url": pdf_url,
                                    "saved_to": str(pdf_destination),
                                    "status": "already_exists",
                                }
                            )
                            continue

                        downloaded_pdfs.append(await download_pdf(http_session, pdf_url, pdf_destination))

                    if depth < config.max_depth:
                        for link in internal_links:
                            if is_same_domain(config.start_url, link) and link not in visited:
                                queue.append((link, depth + 1))

                    crawl_results.append(
                        {
                            "url": current_url,
                            "title": page_title,
                            "pdf_links": pdf_links,
                            "internal_links": internal_links,
                            "page_record": page_result,
                            "downloaded_pdfs": downloaded_pdfs,
                        }
                    )

                except PlaywrightTimeoutError:
                    logging.warning("Timed out while crawling %s", current_url)
                except Exception as exc:  # pragma: no cover - defensive logging only
                    logging.exception("Unexpected crawl failure for %s: %s", current_url, exc)

        await context.close()
        await browser.close()

    index_path = directories["metadata"] / "crawl_index.json"
    await write_json_file(index_path, {"start_url": config.start_url, "results": crawl_results})
    return crawl_results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Async crawler for dynamic government portals")
    parser.add_argument(
        "--start-url",
        default="https://serviceonline.bihar.gov.in/",
        help="Portal landing page to crawl",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent / "documents"),
        help="Directory where fetched documents and page artifacts are stored",
    )
    parser.add_argument(
        "--target-text",
        action="append",
        default=["सामान्य प्रशासन विभाग"],
        help="Accordion text to expand. Provide multiple times for more targets.",
    )
    parser.add_argument("--politeness-delay", type=float, default=2.0, help="Delay between navigations in seconds")
    parser.add_argument("--max-pages", type=int, default=20, help="Maximum number of pages to visit")
    parser.add_argument("--max-depth", type=int, default=2, help="Maximum crawl depth for same-domain links")
    return parser


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_argument_parser().parse_args()

    config = CrawlConfig(
        start_url=args.start_url,
        output_dir=Path(args.output_dir),
        target_texts=args.target_text,
        politeness_delay=args.politeness_delay,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
    )

    results = await crawl_site(config)
    logging.info("Crawl complete. Pages processed: %s", len(results))


if __name__ == "__main__":
    asyncio.run(main())