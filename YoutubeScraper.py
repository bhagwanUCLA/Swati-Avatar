"""
YOUTUBESCRAPER.PY (CLI)
======================
This is a standalone utility for gathering video metadata and transcripts from YouTube.
It is primarily used to collect a list of video URLs from a specific channel or handle.

WORKFLOW:
    1. Run this CLI to get all video links from a channel.
    2. Copy the links from the console or the output file.
    3. Paste them into the main RAG Admin UI for ingestion and indexing.

CLI EXAMPLES:
    python YoutubeScraper.py --handle @Google --max 10 --output links.txt
    python YoutubeScraper.py --channel <ID> --max 5
"""
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")
if not YOUTUBE_API_KEY:
    raise RuntimeError(
        "Missing YOUTUBE_API_KEY. Put it in your .env file or export it in your shell."
    )


class BaseScraper:
    """Base class for all scrapers. Handles rate limiting and caching."""

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_key(self, key: str) -> str:
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, self._cache_key(key) + ".json")

    def _get_cached(self, key: str) -> Optional[dict]:
        path = self._cache_path(key)
        if os.path.exists(path):
            age_hours = (time.time() - os.path.getmtime(path)) / 3600
            if age_hours < 168:  # 7 day cache
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        return None

    def _set_cached(self, key: str, data: dict):
        path = self._cache_path(key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _rate_limit(self, seconds: float = 1.0):
        time.sleep(seconds)


class YouTubeChannelScraper(BaseScraper):
    """
    Uses YouTube Data API + youtube-transcript-api.

    Flow:
      1) channels.list -> uploads playlist
      2) playlistItems.list -> video IDs
      3) videos.list -> metadata
      4) youtube-transcript-api -> transcript
      5) save one JSON per video in videos/
    """

    def __init__(
        self,
        cache_dir: str = "./cache",
        videos_dir: str = "./videos",
        transcript_languages: Optional[List[str]] = None,
        rate_limit_seconds: float = 0.5,
        transcript_workers: int = 4,
    ):
        super().__init__(cache_dir=cache_dir)
        self.videos_dir = Path(videos_dir)
        self.videos_dir.mkdir(parents=True, exist_ok=True)

        self.transcript_languages = transcript_languages or ["en"]
        self.rate_limit_seconds = rate_limit_seconds
        self.transcript_workers = transcript_workers

        print("[INIT] Building YouTube API client...")
        self.youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY, cache_discovery=False)
        self.transcript_api = YouTubeTranscriptApi()
        print("[INIT] YouTube API client ready.")

    def _extract_channel_filter(
        self,
        channel_id: Optional[str] = None,
        handle: Optional[str] = None,
        username: Optional[str] = None,
    ) -> dict:
        filters = [channel_id is not None, handle is not None, username is not None]
        if sum(bool(x) for x in filters) != 1:
            raise ValueError("Specify exactly one of: channel_id, handle, username")

        if channel_id:
            return {"id": channel_id}

        if handle:
            handle = handle if handle.startswith("@") else f"@{handle}"
            return {"forHandle": handle}

        return {"forUsername": username}

    def get_channel_resource(
        self,
        channel_id: Optional[str] = None,
        handle: Optional[str] = None,
        username: Optional[str] = None,
    ) -> dict:
        cache_key = f"channel:{channel_id or handle or username}"
        cached = self._get_cached(cache_key)
        if cached:
            print(f"[CACHE] Channel resource loaded from cache.")
            return cached

        print(f"[API] Calling channels.list for: {channel_id or handle or username}")
        params = {
            "part": "snippet,contentDetails",
            "maxResults": 1,
            **self._extract_channel_filter(
                channel_id=channel_id,
                handle=handle,
                username=username,
            ),
        }

        try:
            response = self.youtube.channels().list(**params).execute()
            print(f"[API] channels.list responded. Items found: {len(response.get('items', []))}")
        except HttpError as e:
            print(f"[ERROR] channels.list HTTP error: {e}")
            raise RuntimeError(f"YouTube channels.list failed: {e}") from e
        except Exception as e:
            print(f"[ERROR] channels.list unexpected error: {e}")
            raise

        # Fallback if handle/username lookup returns nothing
        if not response.get("items") and (handle or username):
            query = handle or username
            print(f"[WARN] No channel found via handle/username. Trying search fallback for: {query}")
            try:
                search_response = self.youtube.search().list(
                    part="snippet",
                    q=query,
                    type="channel",
                    maxResults=5,
                ).execute()
                print(f"[API] search.list responded. Items found: {len(search_response.get('items', []))}")
            except HttpError as e:
                print(f"[ERROR] search.list HTTP error: {e}")
                raise RuntimeError(f"YouTube search.list failed: {e}") from e
            except Exception as e:
                print(f"[ERROR] search.list unexpected error: {e}")
                raise

            channel_ids = []
            for item in search_response.get("items", []):
                ch_id = item.get("id", {}).get("channelId")
                if ch_id:
                    channel_ids.append(ch_id)

            if channel_ids:
                print(f"[API] Re-fetching channel resource by ID: {channel_ids[0]}")
                try:
                    response = self.youtube.channels().list(
                        part="snippet,contentDetails",
                        id=",".join(channel_ids[:1]),
                        maxResults=1,
                    ).execute()
                    print(f"[API] channels.list (by ID) responded. Items found: {len(response.get('items', []))}")
                except HttpError as e:
                    print(f"[ERROR] channels.list (by ID) HTTP error: {e}")
                    raise RuntimeError(f"YouTube channels.list failed: {e}") from e
            else:
                print(f"[ERROR] Fallback search returned no channel IDs for query: {query}")

        self._set_cached(cache_key, response)
        return response

    def get_uploads_playlist_id(self, channel_resource: dict) -> str:
        items = channel_resource.get("items", [])
        if not items:
            print("[ERROR] get_uploads_playlist_id: No items in channel resource.")
            raise ValueError("No channel found for the given filter")

        try:
            playlist_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
            print(f"[INFO] Uploads playlist ID: {playlist_id}")
            return playlist_id
        except KeyError as e:
            print(f"[ERROR] Could not find uploads playlist in channel resource: {e}")
            raise ValueError("Could not find uploads playlist for this channel") from e

    def iter_playlist_video_ids(
        self,
        playlist_id: str,
        max_videos: Optional[int] = None,
        max_pages: int = 50,
    ) -> List[str]:
        video_ids: List[str] = []
        page_token = None
        page_count = 0

        print(f"[INFO] Fetching video IDs from playlist (max_videos={max_videos}, max_pages={max_pages})...")

        while True:
            page_count += 1
            if page_count > max_pages:
                print(f"[WARN] Reached max_pages limit ({max_pages}). Stopping pagination.")
                break

            print(f"[API] Fetching playlist page {page_count}...")
            try:
                response = self.youtube.playlistItems().list(
                    part="snippet,contentDetails",
                    playlistId=playlist_id,
                    maxResults=50,
                    pageToken=page_token,
                ).execute()
            except HttpError as e:
                print(f"[ERROR] playlistItems.list HTTP error on page {page_count}: {e}")
                raise RuntimeError(f"YouTube playlistItems.list failed: {e}") from e
            except Exception as e:
                print(f"[ERROR] playlistItems.list unexpected error on page {page_count}: {e}")
                raise

            page_items = response.get("items", [])
            print(f"[API] Page {page_count} returned {len(page_items)} items.")

            for item in page_items:
                snippet = item.get("snippet", {})
                resource_id = snippet.get("resourceId", {})
                video_id = resource_id.get("videoId")
                if video_id:
                    video_ids.append(video_id)
                    if max_videos and len(video_ids) >= max_videos:
                        print(f"[INFO] Reached max_videos limit ({max_videos}). Stopping.")
                        return video_ids

            page_token = response.get("nextPageToken")
            if not page_token:
                print(f"[INFO] No more pages. Total video IDs collected: {len(video_ids)}")
                break

        return video_ids

    def _chunked(self, items: List[str], size: int = 50):
        for i in range(0, len(items), size):
            yield items[i:i + size]

    def get_video_metadata(self, video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Returns:
          {
            video_id: {
              title, description, date, channel, video_id, video_url
            }
          }
        """
        meta: Dict[str, Dict[str, Any]] = {}
        total_batches = (len(video_ids) + 49) // 50
        print(f"[INFO] Fetching metadata for {len(video_ids)} videos in {total_batches} batch(es)...")

        for batch_num, batch in enumerate(self._chunked(video_ids, 50), start=1):
            print(f"[API] videos.list batch {batch_num}/{total_batches} ({len(batch)} videos)...")
            try:
                response = self.youtube.videos().list(
                    part="snippet",
                    id=",".join(batch),
                ).execute()
            except HttpError as e:
                print(f"[ERROR] videos.list HTTP error on batch {batch_num}: {e}")
                raise RuntimeError(f"YouTube videos.list failed: {e}") from e
            except Exception as e:
                print(f"[ERROR] videos.list unexpected error on batch {batch_num}: {e}")
                raise

            returned = response.get("items", [])
            print(f"[API] Batch {batch_num} returned metadata for {len(returned)} videos.")

            for item in returned:
                snippet = item.get("snippet", {})
                vid = item.get("id")
                if not vid:
                    continue

                published_at = snippet.get("publishedAt", "")
                date_only = published_at[:10] if published_at else None

                meta[vid] = {
                    "video_id": vid,
                    "video_url": f"https://www.youtube.com/watch?v={vid}",
                    "title": snippet.get("title", "") or "",
                    "description": snippet.get("description", "") or "",
                    "date": date_only,
                    "channel": snippet.get("channelTitle", "") or "",
                }

        print(f"[INFO] Metadata fetched for {len(meta)} videos.")
        return meta

    def get_english_transcript(self, video_id: str) -> Optional[str]:
        """
        Returns a single merged transcript string in English.
        Tries English first; if unavailable, tries translation to English.
        """
        try:
            fetched = self.transcript_api.fetch(video_id, languages=self.transcript_languages)
            result = " ".join(s.text for s in fetched).strip()
            print(f"  [TRANSCRIPT] Fetched directly for {video_id} ({len(result)} chars).")
            return result
        except Exception as e:
            print(f"  [TRANSCRIPT] Direct fetch failed for {video_id}: {e}. Trying fallback...")

        try:
            transcript_list = self.transcript_api.list(video_id)

            # Try explicit English transcript
            try:
                transcript = transcript_list.find_transcript(["en"])
                result = " ".join(s.text for s in transcript.fetch()).strip()
                print(f"  [TRANSCRIPT] Found explicit English transcript for {video_id} ({len(result)} chars).")
                return result
            except Exception as e:
                print(f"  [TRANSCRIPT] Explicit English transcript not found for {video_id}: {e}. Trying translation...")

            # Try any translatable transcript and translate to English
            for t in transcript_list:
                try:
                    if getattr(t, "is_translatable", False):
                        result = " ".join(s.text for s in t.translate("en").fetch()).strip()
                        print(f"  [TRANSCRIPT] Translated to English for {video_id} ({len(result)} chars).")
                        return result
                except Exception as e:
                    print(f"  [TRANSCRIPT] Translation attempt failed for {video_id}: {e}")
                    continue
        except Exception as e:
            print(f"  [TRANSCRIPT] All transcript methods failed for {video_id}: {e}")
            return None

        print(f"  [TRANSCRIPT] No transcript available for {video_id}.")
        return None

    def _save_video_json(self, payload: Dict[str, Any]) -> str:
        path = self.videos_dir / f"{payload['video_id']}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return str(path)

    def scrape_channel(
        self,
        channel_id: Optional[str] = None,
        handle: Optional[str] = None,
        username: Optional[str] = None,
        max_videos: Optional[int] = None,
    ) -> List[dict]:
        """
        Scrape a channel's uploads and save one JSON per video in ./videos.
        """
        channel_key = channel_id or handle or username or "unknown"
        cached = self._get_cached(f"channel_scrape:{channel_key}:{max_videos}")
        if cached:
            print(f"[CACHE] Full channel scrape loaded from cache ({len(cached)} videos).")
            return cached

        self._rate_limit(self.rate_limit_seconds)

        print(f"\n[STEP 1/4] Fetching channel resource...")
        channel_resource = self.get_channel_resource(
            channel_id=channel_id,
            handle=handle,
            username=username,
        )

        print(f"\n[STEP 2/4] Resolving uploads playlist ID...")
        uploads_playlist_id = self.get_uploads_playlist_id(channel_resource)

        print(f"\n[STEP 3/4] Collecting video IDs from playlist...")
        video_ids = self.iter_playlist_video_ids(
            uploads_playlist_id,
            max_videos=max_videos,
        )
        print(f"[INFO] Total video IDs collected: {len(video_ids)}")

        print(f"\n[STEP 4/4] Fetching metadata and transcripts for {len(video_ids)} videos...")
        metadata_map = self.get_video_metadata(video_ids)

        results = []
        total = len(video_ids)

        # Sequential by default: simpler, safer, less likely to trip rate limits.
        for idx, vid in enumerate(video_ids, start=1):
            print(f"\n[VIDEO {idx}/{total}] Processing {vid}...")
            meta = metadata_map.get(
                vid,
                {
                    "video_id": vid,
                    "video_url": f"https://www.youtube.com/watch?v={vid}",
                    "title": "",
                    "description": "",
                    "date": None,
                    "channel": "",
                },
            )
            print(f"  [META] Title: {meta.get('title', '(no title)')!r}")

            transcript = self.get_english_transcript(vid) or ""

            payload = {
                "video_id": meta["video_id"],
                "title": meta["title"],
                "description": meta["description"],
                "date": meta["date"],
                "channel": meta["channel"],
                "transcript": transcript,
                "video_url": meta["video_url"],
            }

            try:
                saved_to = self._save_video_json(payload)
                payload["saved_to"] = saved_to
                print(f"  [SAVED] {saved_to}")
            except Exception as e:
                print(f"  [ERROR] Failed to save JSON for {vid}: {e}")
                payload["saved_to"] = None

            results.append(payload)

        print(f"\n[DONE] Scraped {len(results)} videos. Caching results...")
        self._set_cached(f"channel_scrape:{channel_key}:{max_videos}", results)
        return results

    def scrape_video(self, video_id: str) -> dict:
        """
        Scrape a single video using video_id and save JSON.
        """

        cache_key = f"video:{video_id}"
        cached = self._get_cached(cache_key)
        if cached:
            print(f"[CACHE] Video {video_id} loaded from cache.")
            return cached

        self._rate_limit(self.rate_limit_seconds)

        # 1. Get metadata
        print(f"[STEP 1/3] Fetching metadata for video: {video_id}")
        metadata_map = self.get_video_metadata([video_id])
        meta = metadata_map.get(
            video_id,
            {
                "video_id": video_id,
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "title": "",
                "description": "",
                "date": None,
                "channel": "",
            },
        )
        print(f"  [META] Title: {meta.get('title', '(no title)')!r}")

        # 2. Get transcript
        print(f"[STEP 2/3] Fetching transcript for video: {video_id}")
        transcript = self.get_english_transcript(video_id) or ""
        print(f"  [TRANSCRIPT] Length: {len(transcript)} chars")

        # 3. Build payload
        payload = {
            "video_id": meta["video_id"],
            "title": meta["title"],
            "description": meta["description"],
            "date": meta["date"],
            "channel": meta["channel"],
            "transcript": transcript,
            "video_url": meta["video_url"],
        }

        # 4. Save
        print(f"[STEP 3/3] Saving video JSON...")
        try:
            saved_to = self._save_video_json(payload)
            payload["saved_to"] = saved_to
            print(f"  [SAVED] {saved_to}")
        except Exception as e:
            print(f"  [ERROR] Failed to save JSON for {video_id}: {e}")
            payload["saved_to"] = None

        # 5. Cache
        self._set_cached(cache_key, payload)

        return payload


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="YouTube Channel/Video Scraper CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--channel", help="YouTube Channel ID")
    group.add_argument("--handle", help="YouTube Channel Handle (e.g., @Google)")
    group.add_argument("--video", help="Single Video ID")
    
    parser.add_argument("--max", type=int, default=None, help="Maximum number of videos to fetch")
    parser.add_argument("--output", help="Path to save video links to (text file)")
    
    args = parser.parse_args()

    BASE_DIR = Path(__file__).parent
    scraper = YouTubeChannelScraper(
        videos_dir=BASE_DIR / "videos",
        cache_dir=BASE_DIR / "cache",
        transcript_languages=["en"],
        rate_limit_seconds=0.5,
    )

    try:
        if args.video:
            print(f"Scraping single video: {args.video}...")
            result = scraper.scrape_video(args.video)
            print(f"\nTitle: {result['title']}")
            print(f"URL: {result['video_url']}")
            print(f"Saved to: {result['saved_to']}")
            
        else:
            print(f"Scraping channel: {args.channel or args.handle}...")
            results = scraper.scrape_channel(
                channel_id=args.channel,
                handle=args.handle,
                max_videos=args.max
            )
            
            print(f"\nFound {len(results)} videos:")
            links = []
            for r in results:
                print(f"- {r['video_url']} ({r['title']})")
                links.append(r['video_url'])
            
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write("\n".join(links))
                print(f"\nSaved {len(links)} links to {args.output}")
            
            print("\nDone. You can now copy these links into the Ingest UI.")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)