"""
orchestrator.py
---------------
RAGOrchestrator — wires PortfolioScraper, DocumentChunker, FAISSDatabase.

Ingestion methods
-----------------
ingest_portfolio(root_url)            — BFS-crawl a whole site
ingest_section(section_url, name)     — refresh one section
ingest_videos(youtube_urls, section)  — YouTube videos / playlists
ingest_folder(folder_path, section)   — local files (PDF, docx, pptx, txt, md …)
ingest_raw_documents(documents)       — plain-dict list through chunker pipeline

Cache vs Index separation
--------------------------
The scraper cache (raw HTML + Gemini summaries) is independent of FAISS.
- Change chunk_size / model? → rebuild_index() — no HTTP / Gemini calls.
- Add new pages?             → ingest_portfolio() again — cached = free.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

from scraper import PortfolioScraper, ScraperCache, ScrapedDocument, extract_file_with_gemini
from chunker import DocumentChunker, DocumentChunk
from database import FAISSDatabase

logger = logging.getLogger(__name__)

# Extensions handled as plain text (no Gemini needed)
def _is_corrupt_content(text: str, threshold: float = 0.05) -> bool:
    """
    Return True if `text` has a suspiciously high density of low-ASCII
    control characters — the same binary-blob signature that
    `_is_corrupt_html` catches in scraper.py, applied here to extracted
    document content before it enters the index.
    """
    if not text or not text.strip():
        return True
    sample = text[:4_000]
    control = sum(1 for c in sample if ord(c) < 32 and c not in "\t\n\r")
    return (control / max(len(sample), 1)) > threshold


_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".html", ".htm"}

# Extensions that go through Gemini extraction
_BINARY_EXTENSIONS = {
    ".pdf",
    ".docx", ".doc", ".odt",
    ".pptx", ".ppt",
    ".xlsx", ".xls", ".xlsm", ".xlsb", ".csv", ".ods",
}

_VIDEO_EXTENSIONS = {
    ".mp4", ".mpeg", ".mpg", ".mov", ".avi", ".flv", ".webm", ".wmv", ".3gp", ".3gpp",
}

_VIDEO_MIME = {
    ".mp4":  "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg":  "video/mpg",
    ".mov":  "video/quicktime",
    ".avi":  "video/avi",
    ".flv":  "video/x-flv",
    ".webm": "video/webm",
    ".wmv":  "video/wmv",
    ".3gp":  "video/3gpp",
    ".3gpp": "video/3gpp",
}

_ALL_SUPPORTED = _TEXT_EXTENSIONS | _BINARY_EXTENSIONS | _VIDEO_EXTENSIONS


class RAGOrchestrator:
    """
    Parameters
    ----------
    gemini_api_key  : Gemini API key (file/video extraction + embeddings).
    hf_model_name   : Gemini embedding model name (default: "gemini-embedding-001").
    chunk_size      : Characters per chunk.
    chunk_overlap   : Overlap between consecutive chunks.
    dedup_threshold : Cosine similarity threshold for chunk dedup (None = off).
    index_dir       : Directory to save/load the FAISS index.
    cache_dir       : Directory for the scraper disk cache.
    follow_external : Follow links to external domains.
    device          : Torch device for embeddings ("cpu", "cuda", …).
    """

    def __init__(
        self,
        gemini_api_key: str,
        youtube_api_key: Optional[str] = None,
        hf_model_name: str = "gemini-embedding-001",
        chunk_size: int = 3500,
        chunk_overlap: int = 50,
        dedup_threshold: Optional[float] = None,
        min_tokens: int = 20,
        index_dir: str = "./rag_index",
        cache_dir: Optional[str] = None,
        follow_external: bool = True,
        device: str = "cpu",
    ) -> None:

        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.abspath(index_dir)), "scraper_cache"
            )

        self.cache = ScraperCache(cache_dir)

        self.scraper = PortfolioScraper(
            gemini_api_key=gemini_api_key,
            youtube_api_key=youtube_api_key,
            cache=self.cache,
            follow_external=follow_external,
        )

        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            dedup_threshold=dedup_threshold,
            min_tokens=min_tokens,
        )

        self.db = FAISSDatabase(
            model_name=hf_model_name,
            gemini_api_key=gemini_api_key,
            index_path=index_dir if os.path.exists(index_dir) else None,
            device=device,
        )

        self.index_dir = index_dir
        self.min_tokens = min_tokens

        # Separate counter for non-scraper documents (folder / raw).
        # Start high to avoid any overlap with scraper's sequential counter.
        self._aux_doc_counter: int = 900_000

    # ------------------------------------------------------------------
    # Web ingestion
    # ------------------------------------------------------------------

    def ingest_portfolio(self, root_url: str) -> int:
        """Scrape an entire portfolio and store in the vector DB."""
        docs = self.scraper.scrape_portfolio(root_url)
        logger.info("Scraping done: %d docs  (cache: %s)", len(docs), self.cache.stats())

        by_type: dict[str, int] = {}
        by_section: dict[str, int] = {}
        for d in docs:
            by_type[d.doc_type]   = by_type.get(d.doc_type, 0) + 1
            by_section[d.section] = by_section.get(d.section, 0) + 1
        logger.info("  type breakdown:    %s", by_type)
        logger.info("  section breakdown: %s", by_section)

        return self._store_docs(docs)

    def ingest_section(self, section_url: str, section_name: str) -> int:
        """Add or refresh a single section."""
        docs = self.scraper.process_section(section_url, section_name)
        return self._store_docs(docs)

    def ingest_videos(self, youtube_urls: list[str], section: str = "video") -> int:
        """
        Summarise YouTube URLs or playlist URLs via Gemini (cache-aware).

        Playlist URLs (youtube.com/playlist?list=…) are automatically expanded
        to their individual videos before summarisation.
        """
        docs = self.scraper.summarise_videos(youtube_urls, section=section)
        return self._store_docs(docs)

    # ------------------------------------------------------------------
    # Skip-list management
    # ------------------------------------------------------------------

    def list_skipped_urls(self) -> list[str]:
        """Return all URLs currently on the skip list."""
        return self.cache.list_skipped()

    def add_skipped_url(self, url: str) -> None:
        """Manually add a URL to the skip list."""
        self.cache.add_skip(url)

    def remove_skipped_url(self, url: str) -> bool:
        """Remove a URL from the skip list. Returns True if it was present."""
        return self.cache.remove_skip(url)

    def clear_skip_list(self) -> int:
        """Clear the entire skip list. Returns count removed."""
        return self.cache.clear_skip()

    def audit_corrupt_cache(self) -> list[dict]:
        """
        Return a list of cached pages whose HTML is binary/compressed junk.

        Each entry has:
          url          - original requested URL
          final_url    - the actual destination URL (open this manually)
          content_type - Content-Type header stored at scrape time
          cache_file   - path to the .json file on disk

        Use this to find pages that could not be scraped and need to be
        ingested manually via ingest_raw_documents() or ingest_folder().
        """
        return self.cache.find_corrupt_pages()

    def purge_corrupt_cache(self) -> list[dict]:
        """
        Delete all corrupt cache entries and return what was removed.

        After calling this, run ingest_portfolio() again to re-attempt
        those URLs, or ingest them manually with ingest_raw_documents().
        """
        removed = self.cache.delete_corrupt_pages()
        logger.info("purge_corrupt_cache removed %d entries.", len(removed))
        return removed

    def rebuild_index(self, root_url: str) -> int:
        """Re-chunk and re-embed from scraper cache — zero network/API calls."""
        logger.info("Rebuilding index from cache (no network/Gemini calls)...")
        self.scraper.reset()
        docs = self.scraper.scrape_portfolio(root_url)
        logger.info("  %d docs recovered from cache", len(docs))
        self.db.clear()
        return self._store_docs(docs)

    # ------------------------------------------------------------------
    # Local folder ingestion
    # ------------------------------------------------------------------

    def ingest_folder(
        self,
        folder_path: str,
        section: str = "general",
        recursive: bool = True,
    ) -> int:
        """
        Scan a local directory and ingest all supported files.

        Supported file types
        --------------------
        Text  : .txt  .md  .markdown  .rst  .html  .htm
        Binary: .pdf  .docx  .doc  .odt  .pptx  .ppt
                .xlsx  .xls  .xlsm  .xlsb  .csv  .ods
        Video : .mp4  .mpeg  .mpg  .mov  .avi  .flv  .webm  .wmv  .3gp  .3gpp

        Binary and video files are extracted via Gemini (title + content).
        Text files are read directly (title = filename stem).

        Parameters
        ----------
        folder_path : Absolute or relative path to the directory.
        section     : Portfolio section label to assign to all files.
        recursive   : Whether to recurse into sub-directories.
        """
        folder = Path(folder_path).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder_path}")

        pattern = "**/*" if recursive else "*"
        files = sorted(
            f for f in folder.glob(pattern)
            if f.is_file() and f.suffix.lower() in _ALL_SUPPORTED
        )

        if not files:
            logger.warning("ingest_folder: no supported files found in %s", folder)
            return 0

        logger.info("ingest_folder: %d files found in %s", len(files), folder)

        docs: list[ScrapedDocument] = []
        for file_path in files:
            doc = self._process_local_file(file_path, section)
            if doc:
                docs.append(doc)
                logger.info("  [folder] %s → '%s'", file_path.name, doc.title)
            else:
                logger.warning("  [folder] skipped (empty/failed): %s", file_path.name)

        return self._store_docs(docs)

    def _process_local_file(
        self, file_path: Path, section: str
    ) -> Optional[ScrapedDocument]:
        """Convert one local file into a ScrapedDocument."""
        suffix = file_path.suffix.lower()
        url = f"file://{file_path}"

        # ── Plain text / markdown / HTML ─────────────────────────────
        if suffix in _TEXT_EXTENSIONS:
            try:
                raw = file_path.read_text(encoding="utf-8", errors="replace")
                if suffix in (".html", ".htm"):
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(raw, "html.parser")
                    title_tag = soup.find("title")
                    title = (title_tag.get_text(strip=True) or file_path.stem) if title_tag else file_path.stem
                    raw = soup.get_text(separator="\n")
                else:
                    title = file_path.stem
                content = re.sub(r"\n{3,}", "\n\n", raw).strip()
                if not content:
                    logger.warning("  [folder] empty text file: %s", file_path.name)
                    return None
                return self._make_aux_doc(title, section, url, content, "text")
            except Exception as exc:
                logger.warning("Failed to read %s: %s", file_path, exc)
                return None

        # ── Binary / video files — Gemini extraction + local PDF fallback ──
        if suffix in _BINARY_EXTENSIONS or suffix in _VIDEO_EXTENSIONS:
            try:
                import mimetypes as _mt
                file_bytes = file_path.read_bytes()
                if not file_bytes:
                    logger.warning("  [folder] empty file: %s", file_path.name)
                    return None

                # ── Cache hit: skip Gemini if we already extracted this file ─
                if self.cache:
                    cached = self.cache.get_local_file(url)
                    if cached:
                        cached_title = cached["title"] or file_path.stem
                        cached_text  = cached["text"]
                        if cached_text:
                            logger.info("  [folder cache HIT] %s", file_path.name)
                            return self._make_aux_doc(cached_title, section, url, cached_text, "text")

                # Derive an accurate MIME type so Gemini picks the right prompt
                mime_hint = _VIDEO_MIME.get(suffix) or _mt.guess_type(file_path.name)[0] or ""
                if suffix == ".pdf":
                    mime_hint = "application/pdf"

                logger.info("  [folder] Gemini extracting %s (%d KB)", file_path.name, len(file_bytes) // 1024)
                title, content = extract_file_with_gemini(
                    self.scraper.gemini_client,
                    self.scraper.gemini_model,
                    file_bytes,
                    str(file_path),
                    filename=file_path.name,
                    mime_hint=mime_hint,
                    fallback_title=file_path.stem,
                )

                # ── Fallback: local PDF extraction if Gemini returns empty ──
                if not content and suffix == ".pdf":
                    logger.warning(
                        "  [folder] Gemini returned empty for %s — "
                        "falling back to local PDF extraction", file_path.name
                    )
                    from scraper import _pdf_stage1_pypdf, _pdf_stage2_pdfminer
                    content = _pdf_stage1_pypdf(file_bytes) or _pdf_stage2_pdfminer(file_bytes)
                    title = title or file_path.stem

                if not content:
                    logger.warning(
                        "  [folder] all extraction methods returned empty for %s", file_path.name
                    )
                    return None

                # ── Cache the extracted content so re-runs skip Gemini ────────
                if self.cache:
                    self.cache.set_local_file(
                        url=url,
                        title=title or file_path.stem,
                        text=content,
                    )

                return self._make_aux_doc(
                    title or file_path.stem, section, url, content, "text"
                )
            except Exception as exc:
                logger.warning("Failed to extract %s: %s", file_path, exc)
                return None

        return None

    # ------------------------------------------------------------------
    # Raw document ingestion
    # ------------------------------------------------------------------

    def ingest_raw_documents(self, documents: list[dict]) -> int:
        """
        Inject plain-text documents directly through the chunker pipeline.

        Each dict may contain:
          title    (str)  — display title; defaults to "Untitled"
          content  (str)  — full text to be chunked  [required]
          section  (str)  — portfolio section label; defaults to "general"
          url      (str)  — source URL / identifier; defaults to ""
          doc_type (str)  — "text", "index", or "video_summary"; defaults to "text"

        Documents with empty content are silently skipped.
        """
        docs: list[ScrapedDocument] = []
        for item in documents:
            content = (item.get("content") or "").strip()
            if not content:
                continue
            doc = self._make_aux_doc(
                title    = item.get("title", "Untitled"),
                section  = item.get("section", "general"),
                url      = item.get("url", ""),
                content  = content,
                doc_type = item.get("doc_type", "text"),
            )
            docs.append(doc)

        # Auto-register each URL on the skip list so the scraper never
        # re-fetches pages that have been manually ingested.
        for doc in docs:
            if doc.url and doc.url.startswith(("http://", "https://")):
                self.cache.add_skip(doc.url)

        logger.info("ingest_raw_documents: %d valid documents", len(docs))
        return self._store_docs(docs)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = 5,
        section_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
    ) -> list[dict]:
        results = self.db.search(
            question, top_k=top_k * 4, section_filter=section_filter
        )
        if doc_type_filter:
            results = [r for r in results if r.get("doc_type") == doc_type_filter]
        return results[:top_k]

    def query_doc(self, doc_index: int) -> list[DocumentChunk]:
        return self.db.get_all_chunks_for_doc(doc_index)

    # ------------------------------------------------------------------
    # Persistence & diagnostics
    # ------------------------------------------------------------------

    def save(self) -> None:
        self.db.save(self.index_dir)
        logger.info("FAISS index saved → %s", self.index_dir)

    def stats(self) -> dict:
        s = self.db.stats()
        s["cache"] = self.cache.stats()
        return s

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_aux_doc(
        self,
        title: str,
        section: str,
        url: str,
        content: str,
        doc_type: str,
    ) -> ScrapedDocument:
        """Create a ScrapedDocument with a unique aux doc index."""
        self._aux_doc_counter += 1
        return ScrapedDocument(
            index    = self._aux_doc_counter,
            title    = title,
            section  = section,
            url      = url,
            content  = content,
            doc_type = doc_type,
        )

    def _store_docs(self, docs: list[ScrapedDocument]) -> int:
        """
        Chunk → embed → upsert into FAISS, then run short-chunk cleanup.

        Upsert semantics
        ----------------
        For each incoming document whose URL is already in the index, the
        existing chunks are deleted before the new ones are added.  This
        prevents duplicate embeddings from accumulating across re-ingests
        while leaving unrelated documents untouched.
        """
        if not docs:
            return 0

        # ── 1. Corruption guard ───────────────────────────────────────────
        clean_docs = [d for d in docs if not _is_corrupt_content(d.content)]
        skipped = len(docs) - len(clean_docs)
        if skipped:
            logger.warning(
                "_store_docs: dropped %d document(s) with corrupt/empty content.", skipped
            )
            for bad in (d for d in docs if _is_corrupt_content(d.content)):
                logger.warning("  corrupt doc: [%s] %s", bad.section, bad.url or bad.title)
        docs = clean_docs
        if not docs:
            return 0

        # ── 2. Upsert: remove stale chunks for URLs being re-ingested ────
        incoming_urls = {d.url for d in docs if d.url}
        if incoming_urls:
            deleted = self.db.delete_by_urls(incoming_urls)
            if deleted:
                logger.info("_store_docs: upsert removed %d stale chunk(s).", deleted)

        chunks = self.chunker.chunk_documents(
            docs,
            embed_fn=self.db.embed_one if self.chunker.dedup_threshold else None,
        )
        self.db.add(chunks)
        added_count = len(chunks)
        logger.info("Stored %d chunks (before cleanup).", added_count)

        removed_short = self.db.remove_short_chunks(min_tokens=self.min_tokens)
        if removed_short:
            logger.info(
                "Cleanup removed %d short/junk chunks after ingestion.", removed_short
            )

        net_added = max(0, added_count - removed_short)
        logger.info("Net chunks added: %d", net_added)
        return net_added