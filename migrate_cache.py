#!/usr/bin/env python3
"""One-time migration: rename 'summary' -> 'transcript' in scraper_cache/videos/*.json"""
import json
import sys
from pathlib import Path

cache_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./scraper_cache/videos")

if not cache_dir.exists():
    print(f"Directory not found: {cache_dir}")
    sys.exit(0)

updated = 0
for path in sorted(cache_dir.glob("*.json")):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [SKIP] {path.name}: unreadable ({e})")
        continue
    if "summary" in data and "transcript" not in data:
        data["transcript"] = data.pop("summary")
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        updated += 1
        print(f"  [MIGRATED] {path.name}")

print(f"Migrated {updated} file(s) in {cache_dir}")
