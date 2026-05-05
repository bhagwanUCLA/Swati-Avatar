#!/usr/bin/env python3
"""
research_scraper.py — standalone CLI research tool
====================================================
Renamed from ProfileScraper.py. This is NOT the production scraper (see scraper.py).
It is a standalone DuckDuckGo/SerpAPI research CLI used to gather URLs for manual review.

Modes (controlled via CLI):
  --url URL              Scrape a single URL directly.
  --query TOPIC          Use DuckDuckGo via SerpAPI (requires --search) to find
                         up to --max-results URLs, then scrape each one.
  --search               Enable the DuckDuckGo search stage (needed with --query).
  --profile              Use the profile-cleaning Gemini prompt instead of the
                         generic one. Pass when the target is a person's bio page.

Pipeline per URL:
  Stage 0  : DuckDuckGo search via SerpAPI → up to 100 URLs  (search mode only)
  Stage 1  : Crawl root + internal sublinks → raw PageData objects (Playwright)
  Stage 1b : Gemini cleans raw markdown → clean content string
  Stage 2  : LinkJsonBuilder extracts {heading: [links]} map from clean content
  Stage 3  : Assemble ScrapedDocument (title, author, content, links, …) → JSON

Output (written to OUTPUT_DIR/<slug>/):
  profile.md        — cleaned Markdown for the root URL (single-URL mode)
  links.json        — heading → [link] map
  documents.json    — list of ScrapedDocument dicts (all URLs in a search run)
  search_results.json — raw SerpAPI metadata (search mode only)

Usage examples:
  # Single URL, not a profile page
  python scraper.py --url https://example.com/article

  # Single URL, person's profile page
  python scraper.py --url https://www.isb.edu/faculty/deepa-mani --profile

  # DuckDuckGo search, scrape top 30 results, treated as articles
  python scraper.py --query "digital strategy lectures" --search --max-results 30

  # DuckDuckGo search, 100 results, profile prompt
  python scraper.py --query "Sunil Gupta HBS" --search --max-results 100 --profile
"""

import asyncio
import json
import os
import re
import time
import argparse
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse
from pathlib import Path

import aiohttp
import nest_asyncio
import serpapi as serpapi_client
from google import genai
from google.genai import types
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page, Browser, Response as PlaywrightResponse

nest_asyncio.apply()

BASE_DIR = Path(__file__).parent

# ─────────────────────────── CONFIG ──────────────────────────────────────────

# Defaults — all overridable via CLI
DEFAULT_MAX_DEPTH   = 2       # how deep to follow internal links per URL
DEFAULT_MAX_PAGES   = 25      # hard cap on Playwright pages per URL
DEFAULT_MAX_RESULTS = 10      # search results to fetch (max 100)
OUTPUT_DIR          = BASE_DIR / "outputs"

# Gemini
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL    = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

# SerpAPI — DuckDuckGo search
SERPAPI_KEY     = os.environ.get("SERPAPI_KEY", "")   # from serpapi.com
DDG_REGION      = "us-en"               # kl param — e.g. "uk-en", "fr-fr"

# Retry back-off delays (seconds) — used for API exhaustion and Gemini
RETRY_DELAYS    = [5, 15, 30, 60]

# URL path fragments that signal "don't crawl this"
IGNORE_KEYWORDS = [
    "logout", "login", "signin", "signup", "register",
    "privacy", "terms", "cookie", "sitemap", "feed",
    "javascript:", "mailto:", "tel:", "#",
]

# Resource types to block in Playwright (we only want text/data)
BLOCKED_RESOURCES = {"image", "media", "font", "stylesheet"}


# ─────────────────────────── DATA CLASSES ────────────────────────────────────

@dataclass
class LinkEntry:
    """A single hyperlink found in the scraped DOM."""
    name: str
    url: str

    def to_md(self) -> str:
        return f"[{self.name}]({self.url})"


@dataclass
class PageData:
    """Raw data captured for one crawled page before any cleaning."""
    url: str
    heading: str
    content_lines: List[str] = field(default_factory=list)  # DOM in reading order
    links: List[LinkEntry]   = field(default_factory=list)   # all links (for recursion)
    api_chunks: List[dict]   = field(default_factory=list)   # captured XHR/fetch JSON


@dataclass
class SearchResult:
    """One organic result returned by the DuckDuckGo SerpAPI engine."""
    url: str
    title: str
    snippet: str
    displayed_link: str    # e.g. "www.hbs.edu › faculty › sunil-gupta"
    position: int          # rank in the result page (1-based)


@dataclass
class ScrapedDocument:
    """
    Final output for one scraped URL.
    Combines search metadata (if available) with Playwright-scraped content.
    """
    url: str
    title: str
    snippet: str         # from search result or empty string
    displayed_link: str  # from search result or empty string
    position: int        # rank in search results (0 = direct scrape)
    content: str         # cleaned Markdown from Gemini
    links: dict          # {heading: ["[name](url)", ...]}
    source: str          # "search" | "direct"
    is_profile: bool
    scraped_at: str      # ISO 8601 timestamp


# ─────────────────────────── HELPERS ─────────────────────────────────────────

def profile_slug(url: str) -> str:
    """Convert a URL into a filesystem-safe slug for the output directory."""
    p = urlparse(url)
    slug = (p.netloc + p.path).replace("/", "_").replace(".", "_")
    return slug.strip("_") or "profile"


def is_internal(href: str, root: str) -> bool:
    """Return True only if href lives under the root URL's path on the same host."""
    parsed_href = urlparse(href)
    parsed_root = urlparse(root)
    if parsed_href.netloc != parsed_root.netloc:
        return False
    root_path = parsed_root.path.rstrip("/") + "/"
    href_path = parsed_href.path
    return (
        href_path.startswith(root_path)
        or href_path.rstrip("/") == parsed_root.path.rstrip("/")
    )


def should_ignore(url: str) -> bool:
    """Return True if the URL matches any of the IGNORE_KEYWORDS patterns."""
    lower = url.lower()
    return any(kw in lower for kw in IGNORE_KEYWORDS)


def clean_text(text: str) -> str:
    """Strip blank lines and leading/trailing whitespace from every line."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)


# ─────────────────────────── DUCKDUCKGO SEARCHER ─────────────────────────────

class DuckDuckGoSearcher:
    """
    Queries DuckDuckGo via the SerpAPI engine and returns up to max_results
    organic URLs as SearchResult objects.

    DuckDuckGo SerpAPI pagination rules:
      - First request  (start=0)  : returns up to 35 organic results
      - Further pages  (start > 0): returns up to 50 organic results each
    Strategy for 100 results: start=0 (≤35) → start=35 (≤50) → start=85 (≤50)

    Duplicates are removed by URL across pages.
    """

    def __init__(self, api_key: str = SERPAPI_KEY, region: str = DDG_REGION):
        self.api_key = api_key
        self.region  = region

    def _fetch_page(self, query: str, start: int, m: int) -> List[dict]:
        """
        Fetch one page of DuckDuckGo results from SerpAPI.
        Returns the list of organic_results dicts (may be empty on exhaustion).
        `m` is the max number of results to return for this page (1-50).
        """
        client = serpapi_client.Client(api_key=self.api_key)
        try:
            results = client.search({
                "engine": "duckduckgo",
                "q":      query,
                "kl":     self.region,
                "start":  start,
                "m":      m,
            })
            return results.get("organic_results", [])
        except Exception as e:
            print(f"  ! SerpAPI error (start={start}): {e}")
            return []

    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> List[SearchResult]:
        """
        Search DuckDuckGo for `query` and return up to `max_results` SearchResult
        objects, deduplicated by URL.  max_results is capped at 100.
        """
        max_results = min(max_results, 100)
        results:    List[SearchResult] = []
        seen_urls:  Set[str]           = set()
        global_pos = 1   # 1-based position across all pages

        # Page 1: start=0, up to 35 results
        # Page 2+: start=prev_total, up to 50 results each
        offsets = [0] + list(range(35, max_results, 50))

        for start in offsets:
            if len(results) >= max_results:
                break

            remaining = max_results - len(results)
            # First page is capped at 35 by DDG regardless; subsequent at 50
            page_cap = 35 if start == 0 else 50
            m = min(remaining, page_cap)

            print(f"  DDG search: start={start}, m={m} …")
            items = self._fetch_page(query, start=start, m=m)

            if not items:
                print(f"  ! No results returned at start={start} — stopping.")
                break

            for item in items:
                url = item.get("link", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                results.append(SearchResult(
                    url            = url,
                    title          = item.get("title", ""),
                    snippet        = item.get("snippet", ""),
                    displayed_link = item.get("displayed_link", ""),
                    position       = global_pos,
                ))
                global_pos += 1
                if len(results) >= max_results:
                    break

            print(f"  DDG: {len(results)}/{max_results} unique results so far")

        return results


# ─────────────────────────── API EXHAUSTION ──────────────────────────────────

class APIExhauster:
    """
    Replays captured XHR/fetch API calls and follows pagination
    (cursor-based or page/offset-based) with retry + back-off.
    Text-only: silently drops binary/non-JSON responses.
    """

    def __init__(self):
        self._seen_bases: Set[str] = set()

    def _strip_pagination(self, url: str) -> str:
        """Remove known pagination query params to de-dupe API base URLs."""
        return re.sub(r'([?&])(page|offset|cursor|next_cursor|pageToken)=[^&]*', '', url).rstrip("?&")

    def _extract_cursor(self, data: dict) -> Optional[str]:
        """Find a cursor/next-page token in a JSON response envelope."""
        for key in ["next_cursor", "cursor", "nextCursor", "next_page_token", "pageToken", "after"]:
            if val := data.get(key):
                return str(val)
        for sub in ["meta", "pagination", "paging", "links"]:
            node = data.get(sub)
            if isinstance(node, dict):
                for key in ["cursor", "next_cursor", "nextCursor", "next"]:
                    if val := node.get(key):
                        return str(val)
        return None

    def _extract_page_param(self, url: str) -> Optional[tuple]:
        """Return (param_name, current_value) for page/offset params, or None."""
        for param in ["page", "offset"]:
            m = re.search(rf'[?&]{param}=(\d+)', url)
            if m:
                return param, int(m.group(1))
        return None

    def _items_from(self, data: dict) -> list:
        """Extract the result list from common JSON envelope patterns."""
        for key in ["data", "items", "results", "entries", "records"]:
            v = data.get(key)
            if isinstance(v, list):
                return v
        return []

    async def _get(self, session: aiohttp.ClientSession, url: str) -> Optional[dict]:
        """GET a URL with retries; return parsed JSON or None."""
        for delay in RETRY_DELAYS:
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=20),
                    headers={"Accept": "application/json"},
                ) as resp:
                    if resp.status == 200:
                        ct = resp.headers.get("Content-Type", "")
                        if "json" not in ct:
                            return None
                        return await resp.json(content_type=None)
                    elif resp.status in (429, 500, 502, 503, 504):
                        print(f"      ! HTTP {resp.status} → retry in {delay}s")
                        await asyncio.sleep(delay)
                    else:
                        return None
            except asyncio.TimeoutError:
                print(f"      ! Timeout → retry in {delay}s")
                await asyncio.sleep(delay)
            except aiohttp.ClientError as e:
                print(f"      ! ClientError ({e}) → retry in {delay}s")
                await asyncio.sleep(delay)
        print(f"      ! Gave up on {url}")
        return None

    async def exhaust(self, first_url: str, first_response, max_extra_pages: int = 8) -> List[dict]:
        # If the response is a list, treat it as a single chunk and return
        if isinstance(first_response, list):
            return [{"data": first_response, "url": first_url}]

        # Original logic for dict responses follows
        base = self._strip_pagination(first_url)
        if base in self._seen_bases:
            return []
        self._seen_bases.add(base)

        chunks = [first_response]
        cursor = self._extract_cursor(first_response)
        page_info = self._extract_page_param(first_url)

        async with aiohttp.ClientSession() as session:
            if cursor:
                for _ in range(max_extra_pages):
                    sep = "&" if "?" in first_url else "?"
                    next_url = f"{base}{sep}cursor={cursor}"
                    data = await self._get(session, next_url)
                    if data is None:
                        break
                    chunks.append(data)
                    cursor = self._extract_cursor(data)
                    if not cursor:
                        break

            elif page_info:
                param_name, current_val = page_info
                step = current_val if param_name == "offset" and current_val else 1
                for i in range(1, max_extra_pages + 1):
                    next_val = current_val + (i * step if param_name == "offset" else i)
                    next_url = re.sub(
                        rf'([?&]{param_name}=)\d+',
                        lambda m: m.group(1) + str(next_val),
                        first_url,
                    )
                    data = await self._get(session, next_url)
                    if data is None:
                        break
                    if not self._items_from(data):
                        break
                    chunks.append(data)

        return chunks


# ─────────────────────────── CRAWLER ─────────────────────────────────────────

class WebsiteCrawler:
    """
    Playwright-based crawler.
    Starts at a root URL, follows internal sublinks up to MAX_DEPTH /
    MAX_PAGES limits, and captures DOM content + XHR JSON from every page.
    """

    def __init__(self, root_url: str, max_depth: int = DEFAULT_MAX_DEPTH, max_pages: int = DEFAULT_MAX_PAGES):
        self.root_url  = root_url
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited:  Set[str]       = set()
        self.pages:    List[PageData] = []
        self.exhausted = APIExhauster()
        self._browser  = None
        self._pw       = None

    async def start(self):
        """Launch Playwright and open a headless Chromium browser."""
        self._pw      = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=True)

    async def stop(self):
        """Close browser and Playwright cleanly."""
        if self._browser: await self._browser.close()
        if self._pw:      await self._pw.stop()

    async def crawl(self, url: str, depth: int = 0):
        """
        Recursively crawl `url` to `max_depth`, respecting `max_pages`.
        Internal links discovered on each page are queued for recursion.
        """
        url = url.split("#")[0].rstrip("/") or url
        if depth > self.max_depth or len(self.visited) >= self.max_pages:
            return
        if url in self.visited or should_ignore(url):
            return

        self.visited.add(url)
        indent = "  " * depth
        print(f"{indent}[{len(self.visited)}/{self.max_pages}] {url}")

        page_data = await self._scrape(url)
        if page_data:
            self.pages.append(page_data)
            for link in page_data.links:
                if is_internal(link.url, self.root_url) and not should_ignore(link.url):
                    await self.crawl(link.url, depth + 1)

    async def _scrape(self, url: str) -> Optional[PageData]:
        """Open a page in a fresh browser context, expand hidden content, extract DOM."""
        ctx = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )

        await ctx.route(
            "**/*",
            lambda route: (
                route.abort()
                if route.request.resource_type in BLOCKED_RESOURCES
                else route.continue_()
            ),
        )

        captured: List[tuple] = []

        async def on_response(resp: PlaywrightResponse):
            try:
                if resp.request.resource_type not in ("xhr", "fetch"):
                    return
                ct = resp.headers.get("content-type", "")
                if "json" not in ct:
                    return
                body = await resp.json()
                captured.append((resp.url, body))
            except Exception:
                pass

        page = await ctx.new_page()
        page.on("response", on_response)

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            try:
                await page.wait_for_load_state("networkidle", timeout=10_000)
            except Exception:
                pass
            await page.wait_for_timeout(1_500)

            await self._expand(page)

            html  = await page.content()
            soup  = BeautifulSoup(html, "html.parser")

            for tag in soup.find_all(
                ["script", "style", "noscript", "nav", "footer",
                 "img", "svg", "video", "audio", "iframe", "picture"]
            ):
                tag.decompose()

            heading             = self._heading(soup, url)
            content_lines, links = self._extract_content(soup, url)
            api_data            = await self._exhaust_captured(captured)

            return PageData(
                url=url, heading=heading,
                content_lines=content_lines, links=links, api_chunks=api_data,
            )

        except Exception as e:
            print(f"    ! scrape error: {e}")
            return None
        finally:
            await page.close()
            await ctx.close()

    def _heading(self, soup: BeautifulSoup, url: str) -> str:
        """Return the best title-like text from the page (h1 → h2 → <title>)."""
        for tag in ("h1", "h2", "title"):
            el = soup.find(tag)
            if el:
                return el.get_text(strip=True)
        return urlparse(url).path.strip("/").replace("/", " › ") or "Home"

    _PROCESS_TAGS = {"h1","h2","h3","h4","h5","h6","p","li","td","th","blockquote"}
    _HEADING_MAP  = {"h1":"# ","h2":"## ","h3":"### ","h4":"#### ","h5":"##### ","h6":"###### "}
    _INLINE_TAGS  = {"a","strong","em","b","i","span","code","small","label","abbr"}

    def _extract_content(self, soup: BeautifulSoup, base_url: str):
        """
        Single DOM pass in document order.
        Returns (content_lines, links) preserving reading order with inline
        [name](url) links exactly where they appear in the HTML.
        """
        from bs4 import NavigableString, Tag

        seen_urls: Set[str]        = set()
        all_links: List[LinkEntry] = []

        def _process_inline(el) -> str:
            parts = []
            for child in el.children:
                if isinstance(child, NavigableString):
                    t = str(child).strip()
                    if t:
                        parts.append(t)
                elif isinstance(child, Tag):
                    if child.name == "a":
                        href = (child.get("href") or "").strip()
                        text = child.get_text(strip=True)
                        if href:
                            full   = urljoin(base_url, href).split("#")[0]
                            parsed = urlparse(full)
                            if parsed.scheme in ("http", "https"):
                                name = (
                                    text
                                    or child.get("title", "")
                                    or child.get("aria-label", "")
                                    or "Link"
                                )
                                parts.append(f"[{name}]({full})")
                                if full not in seen_urls:
                                    seen_urls.add(full)
                                    all_links.append(LinkEntry(name=name, url=full))
                                continue
                        if text:
                            parts.append(text)
                    elif child.name in self._INLINE_TAGS:
                        parts.append(_process_inline(child))
                    else:
                        t = child.get_text(separator=" ", strip=True)
                        if t:
                            parts.append(t)
            return " ".join(p for p in parts if p)

        def _has_process_ancestor(el) -> bool:
            """True if this element is nested inside another PROCESS_TAG."""
            for parent in el.parents:
                if getattr(parent, "name", None) in self._PROCESS_TAGS:
                    return True
            return False

        content_lines: List[str] = []

        for el in soup.find_all(self._PROCESS_TAGS):
            if _has_process_ancestor(el):
                continue
            if el.name in self._HEADING_MAP:
                text = el.get_text(separator=" ", strip=True)
                if text:
                    content_lines.append(f"{self._HEADING_MAP[el.name]}{text}")
            else:
                line = _process_inline(el).strip()
                if len(line) > 15:
                    content_lines.append(line)

        return content_lines, all_links

    async def _expand(self, page: Page):
        """Click collapsed accordions and force-reveal hidden sections before scraping."""
        try:
            triggers = await page.locator('button[aria-expanded="false"]').all()
            for t in triggers[:20]:
                try:
                    if await t.is_visible():
                        await t.click(force=True, timeout=800)
                        await page.wait_for_timeout(250)
                except Exception:
                    pass

            await page.evaluate("""
                document.querySelectorAll('[aria-expanded="false"]')
                    .forEach(el => el.setAttribute('aria-expanded','true'));
                document.querySelectorAll('[hidden]')
                    .forEach(el => el.removeAttribute('hidden'));
                document.querySelectorAll('[aria-hidden="true"]')
                    .forEach(el => el.setAttribute('aria-hidden','false'));
                document.querySelectorAll(
                    '[class*="accordion__content"],[class*="collapse"],[class*="dropdown__menu"]'
                ).forEach(el => {
                    el.style.display   = 'block';
                    el.style.maxHeight = 'none';
                    el.style.overflow  = 'visible';
                });
            """)
            await page.wait_for_timeout(500)
        except Exception:
            pass

    async def _exhaust_captured(self, captured: List[tuple]) -> List[dict]:
        """Follow pagination for all XHR/fetch calls captured during page load."""
        all_chunks: List[dict] = []
        for api_url, first_json in captured:
            chunks = await self.exhausted.exhaust(api_url, first_json)
            all_chunks.extend(chunks)
        return all_chunks


# ─────────────────────────── MARKDOWN BUILDER ────────────────────────────────

class MarkdownBuilder:
    """Assembles raw PageData objects into a single Markdown string for Gemini."""

    def build_full(self, pages: List[PageData]) -> str:
        """
        Build raw profile.md from all crawled pages.
        Headings → content → inline links → API JSON, one section per page.
        """
        parts: List[str] = []
        for page in pages:
            parts.append(f"# {page.heading}")
            parts.append(f"Source: {page.url}\n")
            parts.extend(page.content_lines)
            parts.append("")
            if page.api_chunks:
                parts.append("## API Data")
                for chunk in page.api_chunks:
                    parts.append("```json")
                    parts.append(json.dumps(chunk, indent=2, ensure_ascii=False))
                    parts.append("```")
                parts.append("")
            parts.append("---\n")
        return "\n".join(parts)


# ─────────────────────────── GEMINI CLEANER ──────────────────────────────────

class GeminiCleaner:
    """
    Sends raw scraped Markdown to Gemini and receives a clean document back.
    Two prompts: one for faculty/person profile pages, one for general content.
    """

    PROFILE_PROMPT = """You are a profile document cleaner.

INPUT: Raw markdown scraped from a person's faculty/portfolio website.
       It may contain navigation menus, site-wide boilerplate, repeated headers,
       index page noise, cookie banners, and other irrelevant content.

TASK: Rewrite it as a single clean, well-structured Markdown profile document.

RULES:
- Keep ALL meaningful content about the person (bio, research, publications,
  awards, courses taught, media, contact info, etc.)
- Keep every [Link name](url) that belongs to actual content — do NOT remove
  or move links; keep them inline exactly where they appear in the text
- Remove: site navigation, breadcrumbs, footer links, "Home / Faculty / ..."
  trails, cookie notices, repeated site-wide headings, index-page card grids
  that just list other people, pagination controls, search bars
- Use clean Markdown headings (## for major sections, ### for sub-sections)
- Do not invent any content that was not in the input
- Return ONLY the cleaned Markdown, no explanation, no fences

INPUT MARKDOWN:
{raw_md}

CLEANED MARKDOWN:"""

    DEFAULT_PROMPT = """You are a Markdown content cleaner.

INPUT: Raw markdown scraped from web pages (articles, posts, blogs, or profiles).
It may include navigation menus, ads, boilerplate, repeated headers, footers,
cookie banners, and other irrelevant UI noise.

TASK: Rewrite it as a clean, well-structured Markdown document.

RULES:
- Keep all meaningful content (main text, headings, sections, lists, quotes)
- Preserve all valid [Link name](url) inline where they appear
- Remove: navigation, breadcrumbs, footers, sidebars, ads, cookie notices,
  pagination, search bars, unrelated links, duplicate/repeated content
- Use clean Markdown structure (## for sections, ### for subsections)
- Maintain logical flow and readability
- Do NOT invent or add new content

OUTPUT:
- Return ONLY cleaned Markdown
- No explanations, no code fences

INPUT:
{raw_md}

OUTPUT:"""

    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def clean(self, raw_md: str, profile: bool = False) -> str:
        """
        Send `raw_md` to Gemini for cleaning.
        `profile=True` uses the faculty/person prompt; False uses the generic one.
        Falls back to the raw markdown if Gemini fails after all retries.
        """
        print("  → Sending to Gemini for cleaning...")
        prompt = (
            self.PROFILE_PROMPT if profile else self.DEFAULT_PROMPT
        ).format(raw_md=raw_md)

        for attempt, delay in enumerate(RETRY_DELAYS, 1):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_level="low")
                    ),
                )
                cleaned = response.text.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```[\w]*\n?", "", cleaned)
                    cleaned = re.sub(r"\n?```$", "", cleaned).strip()
                print(f"  ✓ Gemini returned {len(cleaned):,} chars")
                return cleaned
            except Exception as e:
                if attempt < len(RETRY_DELAYS):
                    print(f"  ! Gemini error ({e}) → retry in {delay}s")
                    time.sleep(delay)
                else:
                    print(f"  ! Gemini gave up after {attempt} attempts: {e}")
                    print("  ! Falling back to raw markdown")
                    return raw_md


# ─────────────────────────── LINK JSON BUILDER ───────────────────────────────

class LinkJsonBuilder:
    """
    Parses cleaned Markdown and extracts all links grouped by their nearest heading.
    Output: {heading_text: ["[Link name](url)", ...], ...}
    Links before the first heading go under "General".
    Pure Python — no LLM needed.
    """

    def build(self, md: str) -> dict:
        """Return a {heading: [link_md, ...]} dict from a Markdown string."""
        result: dict        = {}
        current_heading     = "General"

        for line in md.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                current_heading = stripped.lstrip("#").strip()
                if current_heading not in result:
                    result[current_heading] = []
                continue
            for name, url in re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', stripped):
                if current_heading not in result:
                    result[current_heading] = []
                result[current_heading].append(f"[{name}]({url})")

        return {k: v for k, v in result.items() if v}


# ─────────────────────────── DOCUMENT BUILDER ────────────────────────────────

class DocumentBuilder:
    """
    Orchestrates the per-URL pipeline:
      crawl → build raw markdown → Gemini clean → extract links → ScrapedDocument

    Accepts an optional SearchResult so that search metadata (title, snippet,
    position) can be merged into the final document.
    """

    def __init__(self, max_depth: int = DEFAULT_MAX_DEPTH, max_pages: int = DEFAULT_MAX_PAGES):
        self.max_depth    = max_depth
        self.max_pages    = max_pages
        self.md_builder   = MarkdownBuilder()
        self.gemini       = GeminiCleaner()
        self.link_builder = LinkJsonBuilder()

    async def build(
        self,
        url: str,
        is_profile: bool,
        search_result: Optional[SearchResult] = None,
        keywords: Optional[List[str]] = None,
    ) -> Optional[ScrapedDocument]:
        """
        Run the full pipeline for a single URL and return a ScrapedDocument.
        Returns None if crawling produced no usable content.
        """
        crawler = WebsiteCrawler(url, max_depth=self.max_depth, max_pages=self.max_pages)
        await crawler.start()
        try:
            await crawler.crawl(url)
        finally:
            await crawler.stop()

        if not crawler.pages:
            print(f"  ! No pages scraped from {url}")
            return None

        raw_md   = self.md_builder.build_full(crawler.pages)

        if keywords and not any(k.lower() in raw_md.lower() for k in keywords):
            return None

        clean_md = self.gemini.clean(raw_md, profile=is_profile)
        links    = self.link_builder.build(clean_md)

        # Title: prefer search result title, fall back to first page heading
        title          = (search_result.title          if search_result and search_result.title          else crawler.pages[0].heading)
        snippet        = (search_result.snippet        if search_result and search_result.snippet        else "")
        displayed_link = (search_result.displayed_link if search_result and search_result.displayed_link else "")
        position       = (search_result.position       if search_result else 0)

        return ScrapedDocument(
            url            = url,
            title          = title,
            snippet        = snippet,
            displayed_link = displayed_link,
            position       = position,
            content        = clean_md,
            links          = links,
            source         = "search" if search_result else "direct",
            is_profile     = is_profile,
            scraped_at     = datetime.utcnow().isoformat() + "Z",
        )


# ─────────────────────────── PIPELINE ────────────────────────────────────────

async def run_single(
    url: str,
    is_profile: bool,
    output_dir: Path,
    max_depth: int,
    max_pages: int,
) -> ScrapedDocument:
    """
    Scrape a single URL and write profile.md, links.json, documents.json
    into output_dir/<slug>/.
    """
    slug = profile_slug(url)
    outd = output_dir / slug
    outd.mkdir(parents=True, exist_ok=True)

    builder = DocumentBuilder(max_depth=max_depth, max_pages=max_pages)
    doc = await builder.build(url, is_profile=is_profile)

    if doc:
        (outd / "profile.md").write_text(doc.content, encoding="utf-8")
        (outd / "links.json").write_text(
            json.dumps(doc.links, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (outd / "documents.json").write_text(
            json.dumps([asdict(doc)], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n✓ profile.md   → {outd / 'profile.md'}")
        print(f"✓ links.json   → {outd / 'links.json'}")
        print(f"✓ documents.json → {outd / 'documents.json'}")
    else:
        print("  ! Nothing to save.")

    return doc


async def run_search(
    query: str,
    is_profile: bool,
    output_dir: Path,
    max_results: int,
    max_depth: int,
    max_pages: int,
    region: str,
    keywords: List[str]
) -> List[ScrapedDocument]:
    """
    Search DuckDuckGo via SerpAPI for `query`, scrape each result URL, and
    write all outputs into output_dir/<query_slug>/.

    Produces:
      search_results.json — raw SerpAPI SearchResult metadata
      documents.json      — list of all ScrapedDocument dicts
      links.json          — merged {heading: [links]} across all documents
      001_<slug>/         — per-URL profile.md for individual inspection
    """
    query_slug = re.sub(r"[^\w]+", "_", query).strip("_")
    outd       = output_dir / query_slug
    outd.mkdir(parents=True, exist_ok=True)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"DuckDuckGo Query : {query}")
    print(f"Max results      : {max_results}")
    print(f"Region           : {region}")
    print(f"Output           : {outd}/")
    print(sep + "\n")

    # ── Stage 0: DuckDuckGo search ──
    print("Stage 0 — Searching DuckDuckGo via SerpAPI...\n")
    searcher       = DuckDuckGoSearcher(region=region)
    search_results = searcher.search(query, max_results=max_results)
    print(f"\n✓ Got {len(search_results)} unique search results\n")

    # Save raw search metadata for reference
    search_meta_path = outd / "search_results.json"
    search_meta_path.write_text(
        json.dumps([asdict(r) for r in search_results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Stages 1-3: scrape each URL ──
    builder      = DocumentBuilder(max_depth=max_depth, max_pages=max_pages)
    documents    = []
    merged_links: dict = {}

    for i, sr in enumerate(search_results, 1):
        url = sr.url
        print(f"\n[{i}/{len(search_results)}] (pos={sr.position}) {url}")


        doc = await builder.build(url, is_profile=is_profile, search_result=sr, keywords=keywords)
        if not doc:
            continue
        
        documents.append(doc)

        # Write per-URL profile.md for easy inspection
        url_dir = outd / f"{i:03d}_{profile_slug(url)}"
        url_dir.mkdir(parents=True, exist_ok=True)
        (url_dir / "profile.md").write_text(doc.content, encoding="utf-8")

        # Merge links (heading keys may overlap across docs — extend lists)
        for heading, links in doc.links.items():
            merged_links.setdefault(heading, []).extend(links)

    # ── Write combined outputs ──
    docs_path  = outd / "documents.json"
    links_path = outd / "links.json"

    docs_path.write_text(
        json.dumps([asdict(d) for d in documents], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    links_path.write_text(
        json.dumps(merged_links, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n{sep}")
    print("✓✓✓  COMPLETE")
    print(f"  Documents scraped    : {len(documents)} / {len(search_results)}")
    print(f"  documents.json       → {docs_path}")
    print(f"  links.json           → {links_path}")
    print(f"  search_results.json  → {search_meta_path}")
    print(sep + "\n")

    return documents


# ─────────────────────────── CLI ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Website Scraper — scrape a single URL or a DuckDuckGo search query.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single URL, article
  python scraper.py --url https://example.com/article

  # Single URL, faculty profile
  python scraper.py --url https://www.isb.edu/faculty/deepa-mani --profile

  # DuckDuckGo search, 30 results
  python scraper.py --query "digital transformation lectures" --search --max-results 30

  # DuckDuckGo search, 100 results, profile prompt, UK region
  python scraper.py --query "Sunil Gupta HBS" --search --max-results 100 --profile --region uk-en
        """,
    )
    parser.add_argument("--url",         type=str,
                        help="Single URL to scrape directly.")
    parser.add_argument("--query",       type=str,
                        help="Search topic for DuckDuckGo mode (requires --search).")
    parser.add_argument("--search",      action="store_true",
                        help="Enable DuckDuckGo search via SerpAPI (requires --query).")
    parser.add_argument("--profile",     action="store_true",
                        help="Use the profile-cleaning Gemini prompt (for person/faculty pages).")
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS,
                        help=f"Max search results to fetch and scrape (default {DEFAULT_MAX_RESULTS}, max 100).")
    parser.add_argument("--region",      type=str, default=DDG_REGION,
                        help=f"DuckDuckGo region code (default '{DDG_REGION}', e.g. 'uk-en', 'fr-fr').")
    parser.add_argument("--max-depth",   type=int, default=DEFAULT_MAX_DEPTH,
                        help=f"Internal link crawl depth per URL (default {DEFAULT_MAX_DEPTH}).")
    parser.add_argument("--max-pages",   type=int, default=DEFAULT_MAX_PAGES,
                        help=f"Max Playwright pages opened per URL (default {DEFAULT_MAX_PAGES}).")
    parser.add_argument("--output-dir",  type=str, default=str(OUTPUT_DIR),
                        help=f"Root output directory (default {OUTPUT_DIR}).")

    parser.add_argument(
    "--keywords",
    type=str,
    nargs="+",
    default=[],
    help="Keyword filters used to keep only relevant search results. Example: --keywords ai machine-learning deep-learning",
    )
    return parser.parse_args()

async def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.search:
        if not args.query:
            raise SystemExit("--search requires --query")
        await run_search(
            query=args.query,
            is_profile=args.profile,
            output_dir=output_dir,
            max_results=args.max_results,
            max_depth=args.max_depth,
            max_pages=args.max_pages,
            region=args.region,
            keywords=args.keywords,
        )

    elif args.url:
        await run_single(
            url=args.url,
            is_profile=args.profile,
            output_dir=output_dir,
            max_depth=args.max_depth,
            max_pages=args.max_pages,
        )

    else:
        raise SystemExit(
            "Provide --url for a single scrape, "
            "or --query --search for DuckDuckGo search mode."
        )

if __name__ == "__main__":
    asyncio.run(main())