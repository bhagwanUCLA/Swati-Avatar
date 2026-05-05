"""
delete.py
---------
Interactive CLI for inspecting and cleaning the RAG index and scraper cache.

Usage
-----
    python delete.py                   # interactive menu
    python delete.py --index ./rag_index --cache ./scraper_cache

Main menu
---------
  1  Browse & delete by section
  2  Browse & delete by URL
  3  Browse & delete by document title
  4  Browse & delete by doc_type  (text / index / video_summary)
  5  Find orphaned cache entries  (in cache but NOT in DB)
  6  Find orphaned DB entries     (in DB but NOT in cache)
  7  Purge corrupt cache pages
  8  Manage skip-list
  9  Show stats
 10  Quality filter — delete low-quality chunks
 11  Deduplicate identical chunks by URL
 12  Deduplicate identical chunks by Title
  0  Exit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ── Bootstrap: add project root to path ──────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from database import FAISSDatabase
from scraper import ScraperCache

# ─────────────────────────────────────────────────────────────────────────────
# Colours (graceful degradation on Windows without ANSI)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import colorama
    colorama.init()
    _HAS_COLOR = True
except ImportError:
    _HAS_COLOR = False

def _c(text: str, code: str) -> str:
    if not _HAS_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def _amber(t):  return _c(t, "33")
def _green(t):  return _c(t, "32")
def _red(t):    return _c(t, "31")
def _dim(t):    return _c(t, "2")
def _bold(t):   return _c(t, "1")
def _blue(t):   return _c(t, "34")
def _cyan(t):   return _c(t, "36")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hr(ch: str = "─", width: int = 70) -> None:
    print(_dim(ch * width))


def _yn(prompt: str) -> bool:
    ans = input(f"{prompt} [y/N] ").strip().lower()
    return ans in ("y", "yes")


def _pick(options: list[str], prompt: str = "Choose") -> Optional[int]:
    """Print numbered list, return 0-based index or None for cancel."""
    for i, o in enumerate(options, 1):
        print(f"  {_amber(str(i)):>4}  {o}")
    print(f"  {_dim('  0'):>4}  {_dim('Cancel')}")
    raw = input(f"{prompt}: ").strip()
    if not raw or raw == "0":
        return None
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(options):
            return idx
    except ValueError:
        pass
    print(_red("Invalid choice."))
    return None


def _load_db(index_dir: str, gemini_key: str) -> FAISSDatabase:
    db = FAISSDatabase(gemini_api_key=gemini_key, index_path=index_dir)
    return db


def _save_db(db: FAISSDatabase, index_dir: str) -> None:
    db.save(index_dir)
    print(_green(f"✓ Index saved → {index_dir}"))


# ─────────────────────────────────────────────────────────────────────────────
# Cache introspection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_all_urls(cache: ScraperCache) -> dict[str, dict]:
    """Return {url: cache_entry} for every page in the cache."""
    result: dict[str, dict] = {}
    for path in sorted(cache._pages_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "url" in data:
                result[data["url"]] = data
        except Exception:
            pass
    return result


def _cache_all_videos(cache: ScraperCache) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for path in sorted(cache._videos_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "url" in data:
                result[data["url"]] = data
        except Exception:
            pass
    return result


def _delete_cache_page(cache: ScraperCache, url: str) -> bool:
    import hashlib
    h = hashlib.md5(url.encode()).hexdigest()
    p = cache._pages_dir / f"{h}.json"
    if p.exists():
        p.unlink()
        return True
    return False


def _delete_cache_video(cache: ScraperCache, url: str) -> bool:
    import hashlib
    h = hashlib.md5(url.encode()).hexdigest()
    p = cache._videos_dir / f"{h}.json"
    if p.exists():
        p.unlink()
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# DB introspection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _db_group_by(db: FAISSDatabase, key: str) -> dict[str, list[int]]:
    """Group internal IDs by a chunk metadata field."""
    groups: dict[str, list[int]] = defaultdict(list)
    for iid, chunk in db._meta.items():
        val = getattr(chunk, key, "?") or "?"
        groups[val].append(iid)
    return dict(groups)


def _db_titles_for_group(db: FAISSDatabase, iids: list[int]) -> list[str]:
    titles: set[str] = set()
    for iid in iids:
        c = db._meta.get(iid)
        if c:
            titles.add(c.doc_title or "?")
    return sorted(titles)


def _db_urls_for_group(db: FAISSDatabase, iids: list[int]) -> list[str]:
    urls: set[str] = set()
    for iid in iids:
        c = db._meta.get(iid)
        if c and c.doc_url:
            urls.add(c.doc_url)
    return sorted(urls)


# ─────────────────────────────────────────────────────────────────────────────
# Delete helpers (DB + optionally cache)
# ─────────────────────────────────────────────────────────────────────────────

def _delete_db_iids(db: FAISSDatabase, iids: list[int]) -> int:
    return db._remove_int_ids(iids)


def _ask_also_delete_cache(
    cache: ScraperCache,
    urls: list[str],
) -> None:
    """Ask user whether to also wipe the cache entries for these URLs."""
    if not urls:
        return
    cacheable = [u for u in urls if u.startswith(("http://", "https://", "file://"))]
    if not cacheable:
        return
    print(f"\n  Found {len(cacheable)} URL(s) associated with these DB entries.")
    if _yn("  Also delete their scraper cache entries?"):
        removed_pages  = sum(_delete_cache_page(cache, u) for u in cacheable)
        removed_videos = sum(_delete_cache_video(cache, u) for u in cacheable)
        print(_green(f"  ✓ Removed {removed_pages} page cache + {removed_videos} video cache entries."))


# ─────────────────────────────────────────────────────────────────────────────
# Menu actions
# ─────────────────────────────────────────────────────────────────────────────

def action_by_section(db: FAISSDatabase, cache: ScraperCache, index_dir: str) -> None:
    _hr()
    print(_bold("Delete by section"))
    groups = _db_group_by(db, "section")
    if not groups:
        print(_red("  DB is empty.")); return

    options = [
        f"{_amber(sec):30}  {_dim(str(len(iids)) + ' chunks')}  "
        f"[{_dim(', '.join(_db_titles_for_group(db, iids)[:3]))}{'…' if len(_db_titles_for_group(db, iids)) > 3 else ''}]"
        for sec, iids in sorted(groups.items(), key=lambda x: -len(x[1]))
    ]
    idx = _pick(options, "Select section to delete")
    if idx is None:
        return

    sec  = sorted(groups.keys(), key=lambda s: -len(groups[s]))[idx]
    iids = groups[sec]
    urls = _db_urls_for_group(db, iids)

    print(f"\n  Section : {_amber(sec)}")
    print(f"  Chunks  : {_red(str(len(iids)))}")
    print(f"  Titles  : {', '.join(_db_titles_for_group(db, iids)[:5])}")
    if not _yn(f"\n  Delete ALL {len(iids)} chunks from section '{sec}' from the DB?"):
        print("  Cancelled."); return

    removed = _delete_db_iids(db, iids)
    print(_green(f"  ✓ Removed {removed} chunks from DB."))
    _ask_also_delete_cache(cache, urls)
    _save_db(db, index_dir)


def action_by_url(db: FAISSDatabase, cache: ScraperCache, index_dir: str) -> None:
    _hr()
    print(_bold("Delete by URL"))

    # Build URL → [iids] map
    url_map: dict[str, list[int]] = defaultdict(list)
    for iid, chunk in db._meta.items():
        if chunk.doc_url:
            url_map[chunk.doc_url].append(iid)

    if not url_map:
        print(_red("  No URLs found in DB.")); return

    query = input("  Filter URLs (leave blank for all, or type substring): ").strip()
    filtered = {u: iids for u, iids in url_map.items()
                if not query or query.lower() in u.lower()}
    if not filtered:
        print(_red(f"  No URLs match '{query}'.")); return

    sorted_urls = sorted(filtered.keys())
    options = [
        f"{_blue(u[:80]):85}  {_dim(str(len(filtered[u])) + ' chunks')}"
        for u in sorted_urls
    ]
    idx = _pick(options, "Select URL to delete")
    if idx is None:
        return

    url  = sorted_urls[idx]
    iids = filtered[url]

    print(f"\n  URL    : {_blue(url)}")
    print(f"  Chunks : {_red(str(len(iids)))}")
    print(f"  Title  : {_db_titles_for_group(db, iids)}")

    # Check cache status
    in_page_cache  = cache.get_page(url) is not None
    in_video_cache = cache.get_video(url) is not None
    in_cache = in_page_cache or in_video_cache
    cache_label = _green("YES") if in_cache else _dim("no")
    print(f"  In scraper cache: {cache_label}")

    if not _yn(f"\n  Delete {len(iids)} chunk(s) for this URL from the DB?"):
        print("  Cancelled."); return

    removed = _delete_db_iids(db, iids)
    print(_green(f"  ✓ Removed {removed} chunks from DB."))

    if in_cache and _yn("  Also delete from scraper cache?"):
        p = _delete_cache_page(cache, url)
        v = _delete_cache_video(cache, url)
        print(_green(f"  ✓ Cache: page={'deleted' if p else 'not found'}  video={'deleted' if v else 'not found'}"))

    _save_db(db, index_dir)


def action_by_title(db: FAISSDatabase, cache: ScraperCache, index_dir: str) -> None:
    _hr()
    print(_bold("Delete by document title"))

    title_map: dict[str, list[int]] = defaultdict(list)
    for iid, chunk in db._meta.items():
        title_map[chunk.doc_title or "?"].append(iid)

    query = input("  Search titles (any words, partial match OK): ").strip().lower()

    if query:
        # Score each title by how many query words appear in it
        query_words = query.split()
        def _score(t: str) -> int:
            tl = t.lower()
            return sum(w in tl for w in query_words)
        scored = [(t, _score(t), iids) for t, iids in title_map.items() if _score(t) > 0]
        if not scored:
            print(_red(f"  No titles match \"{query}\".")); return
        # Sort by score desc then alpha; cap at 20
        scored.sort(key=lambda x: (-x[1], x[0]))
        candidates = [(t, iids) for t, _, iids in scored[:20]]
    else:
        candidates = sorted(title_map.items(), key=lambda x: x[0])[:20]

    options = [
        f"{t[:70]:72}  {_dim(str(len(iids)) + ' chunks')}"
        for t, iids in candidates
    ]
    idx = _pick(options, "Select document to delete")
    if idx is None:
        return

    title, iids = candidates[idx]
    urls  = _db_urls_for_group(db, iids)

    print(f"\n  Title  : {_amber(title)}")
    print(f"  Chunks : {_red(str(len(iids)))}")
    print(f"  URL(s) : {', '.join(urls[:3])}")

    if not _yn(f"\n  Delete {len(iids)} chunk(s) for '{title}' from the DB?"):
        print("  Cancelled."); return

    removed = _delete_db_iids(db, iids)
    print(_green(f"  ✓ Removed {removed} chunks from DB."))
    _ask_also_delete_cache(cache, urls)
    _save_db(db, index_dir)


def action_by_doctype(db: FAISSDatabase, cache: ScraperCache, index_dir: str) -> None:
    _hr()
    print(_bold("Delete by doc_type"))
    groups = _db_group_by(db, "doc_type")
    if not groups:
        print(_red("  DB is empty.")); return

    options = [
        f"{_cyan(dt):25}  {_dim(str(len(iids)) + ' chunks')}"
        for dt, iids in sorted(groups.items(), key=lambda x: -len(x[1]))
    ]
    idx = _pick(options, "Select doc_type to delete")
    if idx is None:
        return

    dt   = sorted(groups.keys(), key=lambda d: -len(groups[d]))[idx]
    iids = groups[dt]
    urls = _db_urls_for_group(db, iids)

    print(f"\n  doc_type : {_cyan(dt)}")
    print(f"  Chunks   : {_red(str(len(iids)))}")

    if not _yn(f"\n  Delete ALL {len(iids)} '{dt}' chunks from DB?"):
        print("  Cancelled."); return

    removed = _delete_db_iids(db, iids)
    print(_green(f"  ✓ Removed {removed} chunks from DB."))
    _ask_also_delete_cache(cache, urls)
    _save_db(db, index_dir)


def action_orphan_cache(db: FAISSDatabase, cache: ScraperCache) -> None:
    """Cache entries with NO matching URL in the DB."""
    _hr()
    print(_bold("Orphaned cache entries  (cache present, NOT in DB)"))

    db_urls     = db.get_indexed_urls()
    cache_pages = _cache_all_urls(cache)
    cache_videos= _cache_all_videos(cache)
    all_cache   = set(cache_pages) | set(cache_videos)

    orphans = sorted(all_cache - db_urls)

    if not orphans:
        print(_green("  ✓ No orphaned cache entries found.")); return

    print(f"\n  Found {_red(str(len(orphans)))} orphaned cache URL(s):\n")
    for u in orphans[:50]:
        in_p = _green("page") if u in cache_pages else ""
        in_v = _amber("video") if u in cache_videos else ""
        tags = " ".join(filter(None, [in_p, in_v]))
        print(f"    [{tags}]  {_dim(u[:90])}")
    if len(orphans) > 50:
        print(f"    {_dim(f'… and {len(orphans)-50} more')}")

    print()
    choice = input(
        "  Options:\n"
        "    1  Delete ALL orphaned cache entries\n"
        "    2  Delete a specific URL\n"
        "    0  Cancel\n"
        "  Choice: "
    ).strip()

    if choice == "1":
        if not _yn(f"\n  Delete all {len(orphans)} orphaned cache entries?"):
            print("  Cancelled."); return
        removed = sum(
            _delete_cache_page(cache, u) + _delete_cache_video(cache, u)
            for u in orphans
        )
        print(_green(f"  ✓ Deleted {removed} cache files."))

    elif choice == "2":
        options = [u[:100] for u in orphans]
        idx = _pick(options, "Select URL to delete from cache")
        if idx is None:
            return
        url = orphans[idx]
        p = _delete_cache_page(cache, url)
        v = _delete_cache_video(cache, url)
        print(_green(f"  ✓ page={'deleted' if p else 'not found'}  video={'deleted' if v else 'not found'}"))


def action_orphan_db(db: FAISSDatabase, cache: ScraperCache, index_dir: str) -> None:
    """DB entries with NO matching URL in the cache (file:// URLs ignored)."""
    _hr()
    print(_bold("Orphaned DB entries  (in DB, NOT in cache)"))

    db_urls     = db.get_indexed_urls()
    cache_pages = set(_cache_all_urls(cache).keys())
    cache_videos= set(_cache_all_videos(cache).keys())
    all_cache   = cache_pages | cache_videos

    # Exclude file:// and empty URLs (local-file ingests never go in the page cache)
    web_db_urls = {u for u in db_urls if u.startswith(("http://", "https://"))}
    orphans     = sorted(web_db_urls - all_cache)

    if not orphans:
        print(_green("  ✓ No orphaned DB entries found.")); return

    # Show them grouped with chunk counts
    url_iids: dict[str, list[int]] = defaultdict(list)
    for iid, chunk in db._meta.items():
        if chunk.doc_url in orphans:
            url_iids[chunk.doc_url].append(iid)

    total_chunks = sum(len(v) for v in url_iids.values())
    print(f"\n  Found {_red(str(len(orphans)))} orphaned DB URL(s)  ({total_chunks} chunks total):\n")
    for u in orphans[:50]:
        cnt = len(url_iids.get(u, []))
        print(f"    {_dim(str(cnt) + ' chunks'):12}  {_blue(u[:80])}")
    if len(orphans) > 50:
        print(f"    {_dim(f'… and {len(orphans)-50} more')}")

    print()
    choice = input(
        "  Options:\n"
        "    1  Delete ALL orphaned DB entries\n"
        "    2  Delete a specific URL\n"
        "    0  Cancel\n"
        "  Choice: "
    ).strip()

    if choice == "1":
        if not _yn(f"\n  Delete all {total_chunks} orphaned chunks from DB?"):
            print("  Cancelled."); return
        all_iids = [iid for iids in url_iids.values() for iid in iids]
        removed  = _delete_db_iids(db, all_iids)
        print(_green(f"  ✓ Removed {removed} chunks."))
        _save_db(db, index_dir)

    elif choice == "2":
        options = [f"{u[:80]}  {_dim(str(len(url_iids.get(u,[]))) + ' chunks')}" for u in orphans]
        idx = _pick(options, "Select URL to delete from DB")
        if idx is None:
            return
        url  = orphans[idx]
        iids = url_iids.get(url, [])
        removed = _delete_db_iids(db, iids)
        print(_green(f"  ✓ Removed {removed} chunks."))
        _save_db(db, index_dir)


def action_corrupt_cache(cache: ScraperCache) -> None:
    _hr()
    print(_bold("Purge corrupt cache pages"))
    corrupt = cache.find_corrupt_pages()
    if not corrupt:
        print(_green("  ✓ No corrupt cache entries found.")); return

    print(f"\n  Found {_red(str(len(corrupt)))} corrupt cache entries:\n")
    for e in corrupt[:20]:
        print(f"    {_dim(e['content_type']):30}  {_blue(e['url'][:70])}")
    if len(corrupt) > 20:
        print(f"    {_dim(f'… and {len(corrupt)-20} more')}")

    if _yn(f"\n  Delete all {len(corrupt)} corrupt cache entries?"):
        cache.delete_corrupt_pages()
        print(_green(f"  ✓ Deleted {len(corrupt)} entries."))


def action_skiplist(cache: ScraperCache) -> None:
    _hr()
    print(_bold("Skip-list management"))
    while True:
        skipped = cache.list_skipped()
        print(f"\n  Skip-list has {_amber(str(len(skipped)))} URL(s).\n")
        print("    1  List all")
        print("    2  Add URL")
        print("    3  Remove URL")
        print("    4  Clear all")
        print("    0  Back")
        choice = input("  Choice: ").strip()

        if choice == "0":
            break
        elif choice == "1":
            if not skipped:
                print(_dim("  (empty)"))
            for i, u in enumerate(skipped, 1):
                print(f"    {_dim(str(i)):>5}  {u}")
        elif choice == "2":
            url = input("  URL to add: ").strip()
            if url:
                cache.add_skip(url)
                print(_green(f"  ✓ Added: {url}"))
        elif choice == "3":
            if not skipped:
                print(_dim("  (empty)")); continue
            options = [u[:100] for u in skipped]
            idx = _pick(options, "Select URL to remove")
            if idx is not None:
                removed = cache.remove_skip(skipped[idx])
                print(_green("  ✓ Removed.") if removed else _red("  Not found."))
        elif choice == "4":
            if _yn(f"  Clear all {len(skipped)} skip-list entries?"):
                n = cache.clear_skip()
                print(_green(f"  ✓ Cleared {n} entries."))


def action_stats(db: FAISSDatabase, cache: ScraperCache) -> None:
    _hr()
    print(_bold("Stats"))

    db_stats    = db.stats()
    cache_stats = cache.stats()

    print(f"\n  {_bold('FAISS Index')}")
    print(f"    Total chunks    : {_amber(str(db_stats['total_chunks']))}")
    print(f"    Total documents : {_amber(str(db_stats['total_documents']))}")
    print(f"    Embedding dim   : {db_stats['embedding_dim']}")
    print(f"    Model           : {db_stats['model']}")

    if db_stats.get("sections"):
        print(f"\n  {_bold('Chunks per section')}")
        for sec, cnt in sorted(db_stats["sections"].items(), key=lambda x: -x[1]):
            bar = _amber("█" * min(cnt // max(1, db_stats["total_chunks"] // 30), 30))
            print(f"    {sec:30} {cnt:6}  {bar}")

    print(f"\n  {_bold('Scraper Cache')}")
    print(f"    Cached pages  : {_amber(str(cache_stats['cached_pages']))}")
    print(f"    Cached videos : {_amber(str(cache_stats['cached_videos']))}")

    # Cross-check
    db_urls     = db.get_indexed_urls()
    cache_pages = set(_cache_all_urls(cache).keys())
    cache_videos= set(_cache_all_videos(cache).keys())
    all_cache   = cache_pages | cache_videos
    web_db_urls = {u for u in db_urls if u.startswith(("http://", "https://"))}

    orphan_cache = all_cache - db_urls
    orphan_db    = web_db_urls - all_cache

    status_c = _red(str(len(orphan_cache))) if orphan_cache else _green("0")
    status_d = _red(str(len(orphan_db))) if orphan_db else _green("0")
    print(f"\n  {_bold('Cross-check')}")
    print(f"    Orphaned cache entries (not in DB) : {status_c}")
    print(f"    Orphaned DB URLs (not in cache)    : {status_d}")
    print(f"    Skip-list entries                  : {_amber(str(len(cache.list_skipped())))}")


# ─────────────────────────────────────────────────────────────────────────────
# Main menu
# ─────────────────────────────────────────────────────────────────────────────


def action_quality_filter(db: FAISSDatabase, cache: ScraperCache, index_dir: str) -> None:
    """Delete chunks whose content fails quality heuristics."""
    _hr()
    print(_bold("Quality filter — delete low-quality chunks"))
    print(
        "\n  Scans every chunk and flags ones that look like junk:\n"
        "    • Repeated words  (same word appears N+ times)\n"
        "    • High word-repetition ratio  (repeated words / total words > threshold)\n"
        "    • Very short chunks  (fewer than N tokens)\n"
        "    • Boilerplate patterns  (custom regex)\n"
    )

    # ── Configuration prompts ─────────────────────────────────────────────────
    print(_bold("  Configure filters  (press Enter to skip a filter):\n"))

    def _ask_int(prompt: str, default: int) -> Optional[int]:
        raw = input(f"    {prompt} [default {default}, Enter to skip]: ").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            print(_red("      Invalid — skipping this filter."))
            return None

    def _ask_float(prompt: str, default: float) -> Optional[float]:
        raw = input(f"    {prompt} [default {default:.0%}, Enter to skip]: ").strip()
        if not raw:
            return None
        try:
            v = float(raw.rstrip("%"))
            return v / 100 if v > 1 else v
        except ValueError:
            print(_red("      Invalid — skipping this filter."))
            return None

    min_word_len = _ask_int(
        "Ignore words shorter than N chars when counting repeats (e.g. 4 skips 'the','a','is')", 4
    )
    min_repeats = _ask_int(
        "Delete if any single word repeats at least N times", 5
    )
    ratio_threshold = _ask_float(
        "Delete if (repeated words / total words) exceeds % (e.g. 30)", 0.30
    )
    min_tokens = _ask_int(
        "Delete if chunk has fewer than N words total", 20
    )
    pattern_raw = input(
        "    Custom regex to flag (Enter to skip, e.g. cookie|GDPR): "
    ).strip()
    custom_pattern = None
    if pattern_raw:
        import re as _re
        try:
            custom_pattern = _re.compile(pattern_raw, _re.I)
            print(_green(f"      Pattern compiled: {pattern_raw}"))
        except Exception as exc:
            print(_red(f"      Invalid regex ({exc}) — skipping."))

    if not any([min_word_len, min_repeats, ratio_threshold, min_tokens, custom_pattern]):
        print(_red("  No filters configured — nothing to do.")); return

    # ── Scan all chunks ───────────────────────────────────────────────────────
    import re as _re
    import collections as _col

    flagged: list[tuple[int, str, str]] = []  # (iid, reason, doc_title)

    for iid, chunk in list(db._meta.items()):
        text = (getattr(chunk, "raw_content", "") or getattr(chunk, "text", "") or "").strip()
        if not text:
            flagged.append((iid, "empty content", getattr(chunk, "doc_title", "?")))
            continue

        words = _re.findall(r"\b[a-zA-Z]+\b", text.lower())
        total = len(words)

        # Min tokens filter
        if min_tokens and total < min_tokens:
            flagged.append((iid, f"only {total} words (< {min_tokens})", getattr(chunk, "doc_title", "?")))
            continue

        # Word repetition filters
        if min_word_len or min_repeats or ratio_threshold:
            effective_min_len = min_word_len if min_word_len else 4
            long_words = [w for w in words if len(w) >= effective_min_len]
            counts = _col.Counter(long_words)
            repeated = {w: c for w, c in counts.items() if c >= 2}

            if min_repeats:
                worst_word, worst_cnt = (counts.most_common(1)[0] if counts else ("", 0))
                if worst_cnt >= min_repeats:
                    flagged.append((iid,
                        f"word '{worst_word}' repeats {worst_cnt}x (≥ {min_repeats})",
                        getattr(chunk, "doc_title", "?")))
                    continue

            if ratio_threshold and total > 0:
                repeated_total = sum(c - 1 for c in repeated.values())
                ratio = repeated_total / total
                if ratio >= ratio_threshold:
                    flagged.append((iid,
                        f"repetition ratio {ratio:.0%} (≥ {ratio_threshold:.0%})",
                        getattr(chunk, "doc_title", "?")))
                    continue

        # Custom pattern
        if custom_pattern and custom_pattern.search(text):
            flagged.append((iid, f"matches pattern '{pattern_raw}'", getattr(chunk, "doc_title", "?")))

    if not flagged:
        print(_green("\n  ✓ No chunks flagged — index looks clean.")); return

    # ── Show preview ──────────────────────────────────────────────────────────
    print(f"\n  Found {_red(str(len(flagged)))} chunk(s) to delete:\n")
    # Group by reason for readability
    by_reason: dict[str, list] = _col.defaultdict(list)
    for iid, reason, title in flagged:
        by_reason[reason].append((iid, title))

    for reason, items in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        print(f"    {_amber(str(len(items))):>5}  {_dim(reason)}")
        for _, title in items[:3]:
            print(f"           {title[:70]}")
        if len(items) > 3:
            print(f"           {_dim(f'… and {len(items)-3} more')}")
        print()

    # ── Preview sample chunk ──────────────────────────────────────────────────
    if _yn("  Preview a sample flagged chunk?"):
        iid, reason, title = flagged[0]
        chunk = db._meta[iid]
        text = (getattr(chunk, "raw_content", "") or getattr(chunk, "text", ""))[:400]
        print(f"\n  Title  : {_amber(title)}")
        print(f"  Reason : {_red(reason)}")
        print(f"  Text   :\n{_dim(text)}")
        print()

    if not _yn(f"  Delete all {len(flagged)} flagged chunk(s) from the DB?"):
        print("  Cancelled."); return

    # ── Delete from DB ────────────────────────────────────────────────────────
    iids_to_del = [iid for iid, _, _ in flagged]
    urls_to_del = _db_urls_for_group(db, iids_to_del)
    removed = _delete_db_iids(db, iids_to_del)
    print(_green(f"  ✓ Removed {removed} chunks from DB."))

    # ── Optionally purge cache for affected URLs ───────────────────────────────
    _ask_also_delete_cache(cache, list(urls_to_del))

    _save_db(db, index_dir)


def action_dedup_by_content(db: FAISSDatabase, index_dir: str) -> None:
    _hr()
    print(_bold("Deduplicate identical chunks by content"))

    # Group iids by exact content
    content_groups: dict[str, list[int]] = defaultdict(list)
    for iid, chunk in db._meta.items():
        text = (getattr(chunk, "raw_content", "") or getattr(chunk, "text", "")).strip()
        if text:
            content_groups[text].append(iid)

    duplicates = {text: iids for text, iids in content_groups.items() if len(iids) > 1}
    if not duplicates:
        print(_green("  ✓ No duplicate chunks found."))
        return

    print(f"\n  Found {_red(str(len(duplicates)))} group(s) of identical content.\n")

    all_to_delete = []
    skipped = 0

    for idx, (text, iids) in enumerate(duplicates.items(), 1):
        _hr()
        print(_bold(f"  Group {idx}/{len(duplicates)}  —  {len(iids)} copies\n"))
        print(f"  Content preview:\n  {_dim(text[:200].replace(chr(10), ' '))}\n")

        options = []
        for iid in iids:
            c = db._meta[iid]
            options.append(
                f"section={_amber(c.section or '?')}  "
                f"title={c.doc_title or '?'}  "
                f"url={_dim((c.doc_url or '?')[:60])}"
            )

        print("  Which copy do you want to KEEP?\n")
        keep_idx = _pick(options, "Keep")
        if keep_idx is None:
            print(_dim("  Skipped."))
            skipped += 1
            continue

        to_delete = [iid for i, iid in enumerate(iids) if i != keep_idx]
        all_to_delete.extend(to_delete)
        print(_green(f"  ✓ Will delete {len(to_delete)} copy/copies."))

    if not all_to_delete:
        print(_dim(f"\n  Nothing to delete ({skipped} group(s) skipped)."))
        return

    print(f"\n  Total to delete: {_red(str(len(all_to_delete)))} chunk(s)  ({skipped} group(s) skipped).")
    if not _yn("  Confirm deletion?"):
        print("  Cancelled.")
        return

    removed = _delete_db_iids(db, all_to_delete)
    print(_green(f"  ✓ Removed {removed} duplicate chunks from DB."))
    _save_db(db, index_dir)


def action_dedup(db: FAISSDatabase, index_dir: str, by_key: str) -> None:
    _hr()
    print(_bold(f"Deduplicate chunks by {by_key}"))
    
    # Group by key
    groups = _db_group_by(db, by_key)
    
    total_dups = 0
    to_delete = []
    
    for key_val, iids in groups.items():
        if not key_val or key_val == "?":
            continue
            
        # For this group, find duplicate texts
        seen_texts = set()
        for iid in iids:
            chunk = db._meta[iid]
            text = getattr(chunk, "raw_content", "") or getattr(chunk, "text", "")
            if text in seen_texts:
                to_delete.append(iid)
                total_dups += 1
            else:
                seen_texts.add(text)
                
    if not to_delete:
        print(_green(f"  ✓ No duplicate chunks found when grouping by {by_key}."))
        return
        
    print(f"\n  Found {_red(str(total_dups))} duplicate chunk(s) across {len(groups)} distinct {by_key}(s).")
    
    if not _yn(f"  Delete all {total_dups} duplicate chunks from the DB?"):
        print("  Cancelled.")
        return
        
    removed = _delete_db_iids(db, to_delete)
    print(_green(f"  ✓ Removed {removed} duplicate chunks from DB."))
    _save_db(db, index_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG index & cache manager")
    parser.add_argument("--index", default="./rag_index",   help="Path to FAISS index directory")
    parser.add_argument("--cache", default=None,            help="Path to scraper cache directory (default: sibling of index)")
    parser.add_argument("--key",   default=None,            help="Gemini API key (or set GEMINI_API_KEY env var)")
    args = parser.parse_args()

    index_dir = args.index
    cache_dir = args.cache or str(Path(os.path.abspath(index_dir)).parent / "scraper_cache")
    gemini_key = args.key or os.environ.get("GEMINI_API_KEY", "")

    if not gemini_key:
        # Try loading from .env
        env_path = _HERE / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    gemini_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if not gemini_key:
        print(_red("ERROR: Gemini API key required.  Pass --key or set GEMINI_API_KEY."))
        sys.exit(1)

    print(_bold(_amber("\n  ◈  RAG Manager  ◈\n")))
    print(f"  Index dir : {_dim(index_dir)}")
    print(f"  Cache dir : {_dim(cache_dir)}")
    print()

    # Load DB (no embed call at startup — we only embed on query, not here)
    if not Path(index_dir).exists():
        print(_red(f"ERROR: Index directory not found: {index_dir}"))
        sys.exit(1)

    print("  Loading index…", end="", flush=True)
    db    = _load_db(index_dir, gemini_key)
    cache = ScraperCache(cache_dir)
    print(_green(" done."))

    total = db.stats()["total_chunks"]
    print(f"  {_amber(str(total))} chunks loaded.\n")

    MENU = [
        ("Browse & delete by section",                  lambda: action_by_section(db, cache, index_dir)),
        ("Browse & delete by URL",                      lambda: action_by_url(db, cache, index_dir)),
        ("Browse & delete by document title",           lambda: action_by_title(db, cache, index_dir)),
        ("Browse & delete by doc_type",                 lambda: action_by_doctype(db, cache, index_dir)),
        ("Find orphaned cache entries (not in DB)",     lambda: action_orphan_cache(db, cache)),
        ("Find orphaned DB entries (not in cache)",     lambda: action_orphan_db(db, cache, index_dir)),
        ("Purge corrupt cache pages",                   lambda: action_corrupt_cache(cache)),
        ("Manage skip-list",                            lambda: action_skiplist(cache)),
        ("Show stats",                                  lambda: action_stats(db, cache)),
        ("Quality filter — delete low-quality chunks",  lambda: action_quality_filter(db, cache, index_dir)),
        ("Deduplicate identical chunks by URL",         lambda: action_dedup(db, index_dir, "doc_url")),
        ("Deduplicate identical chunks by Title",        lambda: action_dedup(db, index_dir, "doc_title")),
        ("Deduplicate identical chunks by content",      lambda: action_dedup_by_content(db, index_dir)),
    ]

    while True:
        _hr("═")
        print(_bold("  Main Menu\n"))
        for i, (label, _) in enumerate(MENU, 1):
            print(f"    {_amber(str(i)):>4}  {label}")
        print(f"    {_dim('   0'):>4}  Exit")
        _hr()

        choice = input("  Choice: ").strip()
        if choice == "0":
            print(_dim("  Bye."))
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(MENU):
                MENU[idx][1]()
            else:
                print(_red("  Invalid choice."))
        except ValueError:
            print(_red("  Please enter a number."))
        print()


if __name__ == "__main__":
    main()