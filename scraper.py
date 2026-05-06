"""
SCRAPER.PY
==========
This is the core ingestion engine for the RAG system. It handles two main workflows:

1. WEBSITE CRAWLING (PortfolioScraper + WebsiteCrawler):
   - Uses Playwright to navigate dynamic faculty/portfolio websites.
   - Automatically expands accordions and hidden sections.
   - Routes raw content through GeminiCleaner to produce clean Markdown profile documents.

2. YOUTUBE PROCESSING (_summarise_video):
   - Extracts transcripts using the YouTubeTranscriptApi.
   - Falls back to Gemini's native Video URL feature if transcripts are missing.
   - Formats every video as: [TITLE, URL, SUMMARY/TRANSCRIPT].

USAGE:
    Initialize PortfolioScraper with API keys and call `scrape_portfolio(url)`.
    Results are automatically cached in `./scraper_cache` to prevent redundant API usage.
"""
from __future__ import annotations

import collections
import concurrent.futures
import threading
import hashlib
import io
import json
import logging
import os
import random
import re
import tempfile
import time
from copy import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup, Comment
from google import genai
from google.genai import types

import asyncio
import aiohttp
import nest_asyncio
from playwright.async_api import async_playwright, Page, Browser, Response as PlaywrightResponse
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi

nest_asyncio.apply()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ScrapedDocument:
    """One logical unit of scraped content ready for chunking."""
    index: int
    title: str
    section: str
    url: str
    content: str
    doc_type: str = "text"  # text | index | video_summary
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_YT_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|embed/|shorts/)|youtu\.be/)([\w\-]{6,})",
    flags=re.I,
)

_SKIP_TAGS = {
    "script", "style", "noscript", "head", "meta", "link",
    "svg", "footer", "iframe", "nav", "aside", "header",
}

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language":           "en-US,en;q=0.9",
    "Accept-Encoding":           "gzip, deflate, br",
    "Connection":                "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest":            "document",
    "Sec-Fetch-Mode":            "navigate",
    "Sec-Fetch-Site":            "none",
}

_SECTION_KEYWORDS: dict[str, str] = {
    "biography": "biography", "bio": "biography", "about": "biography", "profile": "biography", "home": "biography",
    "video": "video", "videos": "video", "talks": "video", "presentations": "video", "youtube": "video", "media": "video",
    "opinion": "opinion", "blog": "opinion", "post": "opinion",
    "employment": "employment", "experience": "employment", "career": "employment", "resume": "employment", "cv": "employment",
    "education": "education", "study": "education", "academic": "education",
    "advisor": "advisor", "advisors": "advisor", "advisory": "advisor",
    "cases": "cases", "case": "cases",
    "research": "research", "publications": "research", "papers": "research",
    "associate": "associate_editor", "editor": "associate_editor", "editorial": "associate_editor",
    "contact": "contact", "email": "contact",
    "executive": "executive_teaching", "teaching": "executive_teaching", "courses": "executive_teaching", "mba": "executive_teaching",
    "progress": "research_in_progress", "ongoing": "research_in_progress",
    "working": "working_papers", "workingpapers": "working_papers", "ssrn": "working_papers", "preprint": "working_papers",
    "fame": "fame", "finance": "fame", "accounting": "fame",
}

_IGNORE_KEYWORDS = [
    "logout", "login", "signin", "signup", "register",
    "privacy", "terms", "cookie", "sitemap", "feed",
    "javascript:", "mailto:", "tel:", "#",
]

_BLOCKED_RESOURCES = {"image", "media", "font", "stylesheet"}

_YT_PLAYLIST_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?youtube\.com/(?:playlist|watch)\?(?:[^#]*&)?list=([\w\-]+)",
    flags=re.I,
)


# ---------------------------------------------------------------------------
# Helpers  (single canonical definitions — no duplicates)
# ---------------------------------------------------------------------------

def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_youtube(url: str) -> bool:
    return bool(_YT_PATTERN.search(url or ""))



def _is_pdf_url(url: str) -> bool:
    return bool(url and url.lower().split("?")[0].rstrip("/").endswith(".pdf"))


def _is_pdf_ct(ct: str) -> bool:
    return bool(ct and "application/pdf" in ct.lower())


def _is_office_url(url: str) -> bool:
    if not url:
        return False
    low = url.lower().split("?")[0].rstrip("/")
    return any(low.endswith(ext) for ext in (
        ".xlsx", ".xls", ".xlsm", ".xlsb", ".csv",
        ".docx", ".doc", ".pptx", ".ppt", ".odt", ".ods",
    ))


def _is_office_ct(ct: str) -> bool:
    if not ct:
        return False
    ct = ct.lower()
    office_signatures = [
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "text/csv",
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/vnd.oasis.opendocument.text",
    ]
    return any(sig in ct for sig in office_signatures)


def _clean_text(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _strip_noise(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(_SKIP_TAGS):
        try:
            tag.decompose()
        except Exception:
            tag.extract()
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()


def _extract_plain_text(soup: BeautifulSoup) -> str:
    s = copy(soup)
    _strip_noise(s)
    return _clean_text(s.get_text(separator="\n"))


def _extract_text_with_links(soup: BeautifulSoup, base_url: str) -> str:
    s = copy(soup)
    _strip_noise(s)
    for a_tag in s.find_all("a", href=True):
        try:
            href = urljoin(base_url, a_tag["href"])
        except Exception:
            href = a_tag.get("href", "")
        anchor = a_tag.get_text(strip=True) or href
        a_tag.replace_with(f" [{anchor}]({href}) ")
    return _clean_text(s.get_text(separator="\n"))


def _collect_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    seen: set[str] = set()
    links: list[str] = []
    for a_tag in soup.find_all("a", href=True):
        raw = a_tag["href"].strip()
        if raw.lower().startswith(("javascript:", "mailto:", "tel:", "#", "data:")):
            continue
        try:
            href = urljoin(base_url, raw)
        except Exception:
            continue
        p = urlparse(href)
        if p.scheme not in ("http", "https"):
            continue
        href = urlunparse((p.scheme, p.netloc, p.path, "", p.query, ""))
        if href and href not in seen:
            seen.add(href)
            links.append(href)
    return links


def _url_path_depth(url: str) -> int:
    return len([p for p in urlparse(url).path.strip("/").split("/") if p])


def _first_path_segment(url: str) -> str:
    parts = [p for p in urlparse(url).path.strip("/").split("/") if p]
    return parts[0].lower() if parts else ""


def _infer_section_from_url(url: str, anchor_text: str = "") -> str:
    tokens = re.split(r"[/\-_ .]", (url + " " + anchor_text).lower())
    for token in tokens:
        token = token.strip()
        if token and token in _SECTION_KEYWORDS:
            return _SECTION_KEYWORDS[token]
    return "general"


def _page_title(soup: BeautifulSoup, fallback: str) -> str:
    tag = soup.find("title")
    if tag and tag.string:
        return tag.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return fallback


def _split_gemini_title(text: str, fallback: str) -> tuple[str, str]:
    """
    Parse a TITLE: line that Gemini is instructed to put on line 1.
    Returns (title, content_without_title_line).
    If no TITLE: line is found, returns (fallback, original_text).
    """
    for i, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if stripped.upper().startswith("TITLE:"):
            title = stripped[6:].strip()
            remaining = "\n".join(text.splitlines()[i + 1:]).strip()
            return (title or fallback), remaining
    return fallback, text


def _is_corrupt_html(html: str, threshold: float = 0.05) -> bool:
    """
    Return True when html looks like raw binary/compressed data rather than
    actual HTML text. Heuristic: real HTML is overwhelmingly printable ASCII/UTF-8.
    Binary blobs have a high density of low-ASCII control characters outside
    the normal whitespace set {\\t \\n \\r}.
    """
    if not html:
        return True
    sample = html[:4_000]
    control = sum(1 for c in sample if ord(c) < 32 and c not in "\t\n\r")
    return (control / max(len(sample), 1)) > threshold


# ---------------------------------------------------------------------------
# ScraperCache
# ---------------------------------------------------------------------------

class ScraperCache:
    """
    Layout
    ------
    cache_dir/
      pages/         {md5(url)}.json  -> {url, final_url, html, content_type, cached_at}
      videos/        {md5(url)}.json  -> {url, title, summary, cached_at}
      skip_list.json                  -> sorted list of manually-ingested URLs
    """

    def __init__(self, cache_dir: str = "./scraper_cache") -> None:
        self._pages_dir  = Path(cache_dir) / "pages"
        self._videos_dir = Path(cache_dir) / "videos"
        self._pages_dir.mkdir(parents=True, exist_ok=True)
        self._videos_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Page cache
    # ------------------------------------------------------------------

    def get_page(self, url: str) -> Optional[dict]:
        path = self._pages_dir / f"{_url_hash(url)}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "html" not in data:
                path.unlink(missing_ok=True)
                return None
            return data
        except Exception as exc:
            logger.warning("cache read error (page) %s: %s", url, exc)
            return None

    def set_page(self, url: str, final_url: str, content: str, content_type: str = "text/html") -> None:
        path = self._pages_dir / f"{_url_hash(url)}.json"
        try:
            path.write_text(json.dumps({
                "url":          url,
                "final_url":    final_url,
                "html":         content,
                "content_type": content_type,
                "cached_at":    _now_iso(),
            }, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.warning("cache write error (page) %s: %s", url, exc)

    # ------------------------------------------------------------------
    # Video cache
    # ------------------------------------------------------------------

    def get_video(self, url: str) -> Optional[dict]:
        """Returns {"title": ..., "transcript": ...} or None."""
        path = self._videos_dir / f"{_url_hash(url)}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, str):
                return {"title": "", "transcript": data}
            title   = data.get("title", "")
            content = data.get("transcript") or data.get("summary", "")  # backward compat
            return {"title": title, "transcript": content}
        except Exception as exc:
            logger.warning("cache read error (video) %s: %s", url, exc)
            return None

    def set_video(self, url: str, title: str, transcript: str) -> None:
        path = self._videos_dir / f"{_url_hash(url)}.json"
        try:
            path.write_text(json.dumps({
                "url":        url,
                "title":      title,
                "transcript": transcript,
                "cached_at":  _now_iso(),
            }, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.warning("cache write error (video) %s: %s", url, exc)

    # ------------------------------------------------------------------
    # Local file cache  (dedicated — avoids abusing the page cache format)
    # ------------------------------------------------------------------

    def get_local_file(self, url: str) -> Optional[dict]:
        """Returns {"title": ..., "text": ...} or None."""
        path = self._pages_dir / f"{_url_hash(url)}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("content_type") != "x-local-file":
                return None
            # backward compat: old entries stored title in "final_url", text in "html"
            title = data.get("title") or data.get("final_url", "")
            text  = data.get("text")  or data.get("html", "")
            return {"title": title, "text": text}
        except Exception as exc:
            logger.warning("cache read error (local file) %s: %s", url, exc)
            return None

    def set_local_file(self, url: str, title: str, text: str) -> None:
        path = self._pages_dir / f"{_url_hash(url)}.json"
        try:
            path.write_text(json.dumps({
                "url":          url,
                "title":        title,
                "text":         text,
                "content_type": "x-local-file",
                "cached_at":    _now_iso(),
            }, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.warning("cache write error (local file) %s: %s", url, exc)

    # ------------------------------------------------------------------
    # Skip list  (URLs that have been manually ingested — never re-scrape)
    # ------------------------------------------------------------------

    @property
    def _skip_path(self) -> Path:
        return self._pages_dir.parent / "skip_list.json"

    def _load_skip(self) -> set[str]:
        try:
            if self._skip_path.exists():
                return set(json.loads(self._skip_path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("skip_list read error: %s", exc)
        return set()

    def _save_skip(self, urls: set[str]) -> None:
        try:
            self._skip_path.write_text(
                json.dumps(sorted(urls), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("skip_list write error: %s", exc)

    def add_skip(self, url: str) -> None:
        if not url or not url.startswith(("http://", "https://")):
            return
        s = self._load_skip()
        if url not in s:
            s.add(url)
            self._save_skip(s)
            logger.info("skip_list: added %s", url)

    def remove_skip(self, url: str) -> bool:
        s = self._load_skip()
        if url in s:
            s.discard(url)
            self._save_skip(s)
            return True
        return False

    def is_skipped(self, url: str) -> bool:
        return url in self._load_skip()

    def list_skipped(self) -> list[str]:
        return sorted(self._load_skip())

    def clear_skip(self) -> int:
        s = self._load_skip()
        n = len(s)
        self._save_skip(set())
        return n

    # ------------------------------------------------------------------
    # Corrupt-page audit
    # ------------------------------------------------------------------

    def find_corrupt_pages(self) -> list[dict]:
        corrupt: list[dict] = []
        for path in sorted(self._pages_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                corrupt.append({"url": "?", "final_url": "?",
                                 "content_type": "unreadable-json",
                                 "cache_file": str(path)})
                continue
            if not isinstance(data, dict):
                continue
            if _is_corrupt_html(data.get("html", "")):
                corrupt.append({
                    "url":          data.get("url", "?"),
                    "final_url":    data.get("final_url", "?"),
                    "content_type": data.get("content_type", "?"),
                    "cache_file":   str(path),
                })
        return corrupt

    def delete_corrupt_pages(self) -> list[dict]:
        corrupt = self.find_corrupt_pages()
        for entry in corrupt:
            try:
                Path(entry["cache_file"]).unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("Could not delete %s: %s", entry["cache_file"], exc)
        if corrupt:
            logger.info("Deleted %d corrupt cache entries.", len(corrupt))
        return corrupt

    def stats(self) -> dict:
        return {
            "cached_pages":  len(list(self._pages_dir.glob("*.json"))),
            "cached_videos": len(list(self._videos_dir.glob("*.json"))),
        }


# ---------------------------------------------------------------------------
# GeminiCleaner
# ---------------------------------------------------------------------------

class GeminiCleaner:
    def __init__(self, client: genai.Client, model: str):
        self.client = client
        self.model  = model
        self.prompt = (
            "You are a profile document cleaner.\n"
            "INPUT: Raw markdown scraped from a faculty/portfolio website.\n"
            "TASK: Rewrite it as a single clean, well-structured Markdown profile document.\n"
            "Keep all meaningful bio/research/pubs. Remove navigation/footers.\n"
        )

    def clean(self, raw_md: str) -> str:
        if not self.client:
            return raw_md
        try:
            res = self.client.models.generate_content(
                model=self.model, contents=self.prompt + raw_md
            )
            return res.text.strip()
        except Exception as e:
            logger.warning("Gemini cleaner failed: %s", e)
            return raw_md


# ---------------------------------------------------------------------------
# Data class for crawler pages
# ---------------------------------------------------------------------------

@dataclass
class PageData:
    url: str
    heading: str
    content_lines: list[str]
    links: list[dict]


# ---------------------------------------------------------------------------
# WebsiteCrawler  — FIX: single browser instance reused across all pages
# ---------------------------------------------------------------------------

class WebsiteCrawler:
    def __init__(self, root_url: str, max_pages: int = 50, delay: float = 0.5, cache: Optional["ScraperCache"] = None):
        self.root_url  = root_url
        self.max_pages = max_pages
        self.delay     = delay
        self._cache    = cache
        self.visited:  set[str]       = set()
        self.pages:    list[PageData] = []
        self._browser  = None
        self._pw       = None

    async def crawl(self, url: str, max_depth: int = 2, current_depth: int = 0):
        url = url.split("#")[0].rstrip("/")
        if current_depth > max_depth or len(self.visited) >= self.max_pages or url in self.visited:
            return

        self.visited.add(url)

        # ── Cache hit: reconstruct PageData from stored HTML ─────────────
        if self._cache:
            cached = self._cache.get_page(url)
            if cached and not _is_corrupt_html(cached.get("html", "")):
                logger.info("  [cache HIT] %s", url)
                soup = BeautifulSoup(cached["html"], "html.parser")
                for s in soup(["script", "style", "nav", "footer", "header"]):
                    s.decompose()
                heading = (soup.title.get_text(strip=True) or url) if soup.title else url
                lines   = [
                    p.get_text(strip=True)
                    for p in soup.find_all(["p", "h1", "h2", "h3", "li"])
                    if p.get_text(strip=True)
                ]
                links = []
                for a in soup.find_all("a", href=True):
                    href = urljoin(url, a["href"]).split("#")[0]
                    if urlparse(href).netloc == urlparse(self.root_url).netloc:
                        links.append({"url": href})
                self.pages.append(PageData(url=url, heading=heading, content_lines=lines, links=links))
                for link in links:
                    await self.crawl(link["url"], max_depth, current_depth + 1)
                if current_depth == 0 and self._browser:
                    await self._browser.close()
                    await self._pw.stop()
                    self._browser = None
                    self._pw      = None
                return

        # ── Launch browser once on the very first Playwright fetch ───────
        if self._pw is None:
            self._pw      = await async_playwright().start()
            self._browser = await self._pw.chromium.launch(headless=True)

        logger.info("  [Playwright] %s", url)
        page = await self._browser.new_page(user_agent=_DEFAULT_HEADERS["User-Agent"])
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(self.delay)
            await page.evaluate("""
                document.querySelectorAll('[aria-expanded="false"]').forEach(el => el.click());
                document.querySelectorAll('[hidden]').forEach(el => el.removeAttribute('hidden'));
                document.querySelectorAll('[aria-hidden="true"]').forEach(el => el.setAttribute('aria-hidden','false'));
                document.querySelectorAll('[class*="accordion__content"],[class*="collapse"],[class*="dropdown__menu"]')
                    .forEach(el => {
                        el.style.display   = 'block';
                        el.style.maxHeight = 'none';
                        el.style.overflow  = 'visible';
                    });
            """)

            content = await page.content()
            if self._cache:
                self._cache.set_page(url, final_url=url, content=content, content_type="text/html")

            soup = BeautifulSoup(content, "html.parser")
            for s in soup(["script", "style", "nav", "footer", "header"]):
                s.decompose()

            heading = (soup.title.get_text(strip=True) or url) if soup.title else url
            lines   = [
                p.get_text(strip=True)
                for p in soup.find_all(["p", "h1", "h2", "h3", "li"])
                if p.get_text(strip=True)
            ]

            links = []
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"]).split("#")[0]
                if urlparse(href).netloc == urlparse(self.root_url).netloc:
                    links.append({"url": href})

            self.pages.append(PageData(url=url, heading=heading, content_lines=lines, links=links))

            for link in links:
                await self.crawl(link["url"], max_depth, current_depth + 1)

        except Exception as e:
            logger.error("Crawl error %s: %s", url, e)
        finally:
            await page.close()
            # Close browser only after the top-level call (depth 0) finishes
            if current_depth == 0 and self._browser:
                await self._browser.close()
                await self._pw.stop()
                self._browser = None
                self._pw      = None


# ---------------------------------------------------------------------------
# YouTubeChannelScraper  — FIX: s.text not s["text"] for v1.x API
# ---------------------------------------------------------------------------

class YouTubeChannelScraper:
    def __init__(self, api_key: Optional[str]):
        self.youtube        = build("youtube", "v3", developerKey=api_key, cache_discovery=False) if api_key else None
        self.transcript_api = YouTubeTranscriptApi()

    def get_uploads_playlist_id(self, channel_url: str) -> Optional[str]:
        if not self.youtube:
            return None
        try:
            cid    = None
            handle = None
            if "/channel/" in channel_url:
                cid    = channel_url.split("/channel/")[1].split("/")[0]
            elif "/@" in channel_url:
                handle = "@" + channel_url.split("/@")[1].split("/")[0]

            params = {"part": "contentDetails", "maxResults": 1}
            if cid:    params["id"]        = cid
            elif handle: params["forHandle"] = handle
            else:        return None

            res = self.youtube.channels().list(**params).execute()
            if res.get("items"):
                return res["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        except Exception:
            pass
        return None

    def get_playlist_video_ids(self, playlist_id: str, max_videos: int = 50) -> list[str]:
        if not self.youtube:
            return []
        try:
            res = self.youtube.playlistItems().list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=min(max_videos, 50),
            ).execute()
            return [i["contentDetails"]["videoId"] for i in res.get("items", [])]
        except Exception:
            return []

    def get_video_title(self, video_id: str) -> Optional[str]:
        if not self.youtube:
            return None
        try:
            res = self.youtube.videos().list(part="snippet", id=video_id).execute()
            if res.get("items"):
                return res["items"][0]["snippet"]["title"]
        except Exception:
            pass
        return None

    def get_transcript(self, video_id: str) -> Optional[str]:
        # FIX: use s.text (attribute) not s["text"] (dict) — youtube-transcript-api v1.x
        try:
            fetched = self.transcript_api.fetch(video_id, languages=["en"])
            return " ".join([s.text for s in fetched]).strip()
        except Exception:
            pass

        try:
            t_list = self.transcript_api.list(video_id)
            # Try explicit English first
            try:
                transcript = t_list.find_transcript(["en"])
                return " ".join([s.text for s in transcript.fetch()]).strip()
            except Exception:
                pass
            # Fall back to any translatable transcript
            for t in t_list:
                try:
                    if t.is_translatable:
                        return " ".join([s.text for s in t.translate("en").fetch()]).strip()
                except Exception:
                    continue
        except Exception:
            pass

        return None


# ---------------------------------------------------------------------------
# PortfolioScraper
# ---------------------------------------------------------------------------

class PortfolioScraper:
    def __init__(
        self,
        gemini_api_key: str,
        youtube_api_key: Optional[str] = None,
        cache: Optional[ScraperCache] = None,
        gemini_model: str = "gemini-3-flash-preview",
        max_crawl_pages: int = 50,
        request_delay: float = 0.5,
        **kwargs,
    ):
        self.gemini_model    = gemini_model
        self.max_crawl_pages = max_crawl_pages
        self.delay           = request_delay
        self.cache           = cache

        try:
            self.gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None
        except Exception:
            self.gemini_client = None

        self._cleaner     = GeminiCleaner(self.gemini_client, gemini_model)
        self._yt_scraper  = YouTubeChannelScraper(youtube_api_key)
        self._doc_counter = 0
        self._lock        = threading.Lock()

    # ------------------------------------------------------------------
    # Public scraping entry points
    # ------------------------------------------------------------------

    def scrape_portfolio(self, root_url: str) -> list[ScrapedDocument]:
        root_url = root_url.rstrip("/")

        # 1. YouTube channel, playlist, or single video
        if "youtube.com" in root_url or _is_youtube(root_url):
            return self._expand_to_video_docs(root_url, "video")

        # 2. Profile website
        crawler = WebsiteCrawler(root_url, max_pages=self.max_crawl_pages, delay=self.delay, cache=self.cache)
        asyncio.run(crawler.crawl(root_url))

        docs         = []
        raw_md_parts = []
        for p in crawler.pages:
            txt = "\n".join(p.content_lines)
            raw_md_parts.append(f"## {p.heading}\nURL: {p.url}\n\n{txt}")
            docs.append(self._make_doc(p.heading, _infer_section_from_url(p.url), p.url, txt, "text"))

        full_raw = "\n\n".join(raw_md_parts)
        cleaned  = self._cleaner.clean(full_raw)
        if len(cleaned) > 100:
            docs.append(
                self._make_doc("Full Profile Summary", "biography", root_url, cleaned, "text", {"is_cleaned": True})
            )

        return docs

    def process_section(self, url: str, name: str) -> list[ScrapedDocument]:
        crawler = WebsiteCrawler(url, max_pages=1, delay=self.delay, cache=self.cache)
        asyncio.run(crawler.crawl(url, max_depth=0))
        if not crawler.pages:
            return []
        p = crawler.pages[0]
        return [self._make_doc(p.heading, name, p.url, "\n".join(p.content_lines), "text")]

    def summarise_videos(self, urls: list[str], section: str = "video") -> list[ScrapedDocument]:
        docs = []
        for u in urls:
            docs.extend(self._expand_to_video_docs(u, section))
        return docs

    # ------------------------------------------------------------------
    # Internal: video helpers
    # ------------------------------------------------------------------

    def _expand_to_video_docs(self, url: str, section: str) -> list[ScrapedDocument]:
        """Expand a channel/playlist URL to per-video docs, or handle a single video."""
        if "youtube.com" in url and ("/channel/" in url or "/@" in url or "list=" in url):
            if "list=" in url:
                pid = urlparse(url).query.split("list=")[1].split("&")[0]
            else:
                pid = self._yt_scraper.get_uploads_playlist_id(url)
            if pid:
                vids = self._yt_scraper.get_playlist_video_ids(pid, self.max_crawl_pages)
                return [doc for v in vids
                        for doc in self._summarise_video(f"https://www.youtube.com/watch?v={v}", section)]
            return []
        return self._summarise_video(url, section)

    def _summarise_video(self, url: str, section: str) -> list[ScrapedDocument]:
        if self.cache:
            c = self.cache.get_video(url)
            if c:
                return [self._make_doc(c["title"], section, url, c["transcript"], "video_summary")]

        match = _YT_PATTERN.search(url)
        vid = match.group(1) if match else url.split("/")[-1].split("?")[0]
        title      = self._yt_scraper.get_video_title(vid) or f"Video {vid}"
        transcript = self._yt_scraper.get_transcript(vid)

        if transcript:
            if self.cache:
                self.cache.set_video(url, title, transcript)
            return [self._make_doc(title, section, url, transcript, "video_summary")]

        # Fallback: ask Gemini to summarise the video directly via its URL
        if not self.gemini_client:
            return []
        try:
            res = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=[
                    types.Part(
                        file_data=types.FileData(
                            file_uri=url,
                            mime_type="video/*",   # ✅ required for YouTube URLs
                        )
                    ),
                    types.Part(                    # ✅ must be a Part, not a plain string
                        text="Provide TITLE: <title> on the first line, then the full transcript or summary on the following lines."
                    ),
                ],
            )
            t, s = _split_gemini_title(res.text, title)
            if self.cache:
                self.cache.set_video(url, t, s)
            return [self._make_doc(t, section, url, s, "video_summary")]
        except Exception as e:
            logger.warning("Gemini video fallback failed for %s: %s", url, e)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_doc(self, title, section, url, content, doc_type, extra=None) -> ScrapedDocument:
        with self._lock:
            self._doc_counter += 1
            idx = self._doc_counter
        return ScrapedDocument(idx, title, section, url, content, doc_type, extra or {})

    def reset(self):
        with self._lock:
            self._doc_counter = 0


# ---------------------------------------------------------------------------
# Gemini file extraction  (module-level — upload → generate → delete)
# ---------------------------------------------------------------------------

def extract_file_with_gemini(
    gemini_client,
    model: str,
    file_bytes: bytes,
    source_url: str,
    filename: Optional[str] = None,
    mime_hint: Optional[str] = None,
    fallback_title: str = "",
) -> tuple[str, str]:
    """
    Upload a file to the Gemini Files API, extract structured content, then delete it.

    Every prompt instructs Gemini to output:
        TITLE: <document title>
        <rest of extracted content>

    Returns (title, content). Both are empty strings on failure.
    """
    if not gemini_client:
        logger.warning("Gemini client not configured; cannot extract file: %s", source_url)
        return fallback_title, ""

    tmp_path: Optional[str] = None
    uploaded = None
    try:
        suffix = ""
        if filename and "." in filename:
            suffix = "." + filename.split(".")[-1]
        elif source_url and "." in source_url.split("/")[-1]:
            suffix = "." + source_url.split("/")[-1].split("?")[0].split(".")[-1]

        fd, tmp_path = tempfile.mkstemp(suffix=suffix or "")
        os.close(fd)
        with open(tmp_path, "wb") as fh:
            fh.write(file_bytes)

        logger.info("  [Gemini file] uploading %s (%d KB)", source_url, len(file_bytes) // 1024)
        uploaded = gemini_client.files.upload(file=tmp_path)

        import time
        while True:
            file_info = gemini_client.files.get(name=uploaded.name)
            state_str = str(file_info.state).upper()
            if "PROCESSING" in state_str:
                logger.info("  [Gemini file] Status: %s. Retrying in 5 seconds...", state_str)
                time.sleep(5)
            elif "FAILED" in state_str:
                logger.error("  [Gemini file] File processing failed for %s", source_url)
                return fallback_title, ""
            else:
                break

        ext       = (filename or source_url or "").lower().split("?")[0]
        mime_type = mime_hint or ""

        if ext.endswith(".pdf") or (mime_hint and "pdf" in mime_hint):
            mime_type = "application/pdf"
            prompt = (
                "You are extracting content from a PDF document.\n\n"
                "Output format (follow exactly):\n"
                "TITLE: <the document's actual title>\n\n"
                "<full extracted text — preserve headings, paragraphs, tables, bullet lists; "
                "use LaTeX for mathematical formulas; do not summarise or omit any content>"
            )
        elif (ext.endswith((".xlsx", ".xls", ".xlsm", ".xlsb", ".csv"))
              or (mime_hint and "spreadsheet" in mime_hint)
              or ext.endswith(".ods")):
            prompt = (
                "You are extracting content from a spreadsheet.\n\n"
                "Output format (follow exactly):\n"
                "TITLE: <the spreadsheet's name or main topic>\n\n"
                "<for each sheet, output a markdown table or CSV block with the sheet name as a heading>"
            )
        elif ext.endswith((".docx", ".doc", ".odt")) or (mime_hint and "word" in mime_hint):
            prompt = (
                "You are extracting content from a Word document.\n\n"
                "Output format (follow exactly):\n"
                "TITLE: <the document's actual title or subject>\n\n"
                "<full text — preserve headings, paragraphs, lists and tables; do not summarise>"
            )
        elif ext.endswith((".pptx", ".ppt")) or (mime_hint and "presentation" in mime_hint):
            prompt = (
                "You are extracting content from a presentation.\n\n"
                "Output format (follow exactly):\n"
                "TITLE: <the presentation's title>\n\n"
                "<slide-by-slide content: slide number, title and bullet points>"
            )
        elif (mime_hint and "video" in mime_hint) or ext.endswith(
            (".mp4", ".mpeg", ".mpg", ".mov", ".avi", ".flv", ".webm", ".wmv", ".3gp", ".3gpp")
        ):
            prompt = (
                "You are extracting content from a video file.\n\n"
                "Output format (follow exactly):\n"
                "TITLE: <the video's title or main topic>\n\n"
                "<full transcript or detailed summary — include all spoken content, key points, and topics discussed>"
            )
        else:
            prompt = (
                "You are extracting content from a document.\n\n"
                "Output format (follow exactly):\n"
                "TITLE: <the document's actual title or main topic>\n\n"
                "<full extracted content — preserve headings, paragraphs, lists and tables>"
            )

        raw_text = ""
        try:
            contents = types.Content(parts=[
                types.Part(file_data=types.FileData(
                    file_uri=getattr(uploaded, "uri", getattr(uploaded, "name", None)),
                    mime_type=mime_type or None,
                )),
                types.Part(text=prompt),
            ])
            response = gemini_client.models.generate_content(model=model, contents=contents)
            raw_text = getattr(response, "text", "") or ""
        except Exception as exc:
            logger.warning("generate_content(parts=…) failed; trying fallback: %s", exc)
            try:
                response = gemini_client.models.generate_content(
                    model=model,
                    contents=[uploaded, prompt],
                )
                raw_text = getattr(response, "text", "") or ""
            except Exception as exc2:
                logger.error("Gemini generate_content failed for %s: %s", source_url, exc2)

        # Best-effort delete uploaded file
        try:
            if getattr(uploaded, "name", None) and hasattr(gemini_client.files, "delete"):
                gemini_client.files.delete(name=uploaded.name)
        except Exception:
            pass

        if not raw_text:
            return fallback_title, ""

        title, content = _split_gemini_title(raw_text, fallback=fallback_title)
        return title, _clean_text(content)

    except Exception as exc:
        logger.error("Gemini file extraction failed [%s]: %s", source_url, exc)
        return fallback_title, ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Optional local PDF fallback helpers (used by orchestrator)
# ---------------------------------------------------------------------------

def _pdf_stage1_pypdf(b: bytes) -> str:
    try:
        from pypdf import PdfReader
        r = PdfReader(io.BytesIO(b))
        return "\n".join([p.extract_text() for p in r.pages])
    except Exception:
        return ""


def _pdf_stage2_pdfminer(b: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text
        return extract_text(io.BytesIO(b))
    except Exception:
        return ""