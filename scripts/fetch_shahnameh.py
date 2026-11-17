"""Download Ferdowsi's Shahnameh from the Ganjoor API and write it as plain text.

Output format (data/shahnameh.txt):
    one couplet (bayt) per line, the two hemistichs (mesra) separated by " / ",
    sections (Ganjoor "bakhsh" pages) separated by a blank line.

A machine-readable index (data/sections.json) records, for every section, its
Ganjoor id, its story (e.g. "Sohrab") and its line range in shahnameh.txt, so
the train/held-out split can be made at section level and audited later.

Usage:
    python scripts/fetch_shahnameh.py            # ~1000 polite requests, ~5 min
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

API = "https://api.ganjoor.net/api/ganjoor"
ROOT_CAT = 33  # /ferdousi/shahname
OUT = Path(__file__).resolve().parent.parent / "data"
POEM_QUERY = (
    "catInfo=false&catPoems=false&rhymes=false&recitations=false&images=false"
    "&songs=false&comments=false&verseDetails=true&navigation=false&relatedpoems=false"
)


def get(url: str, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "minigpt-corpus-fetch"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:  # network hiccups: back off and retry
            if attempt == retries - 1:
                raise
            wait = 2**attempt
            print(f"  retry in {wait}s ({exc})")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def clean(text: str) -> str:
    # Normalise whitespace; keep the zero-width non-joiner (U+200C), which is
    # orthographically meaningful in Persian (e.g. "می‌رود").
    return " ".join(text.replace(" ", " ").split())


def main() -> None:
    OUT.mkdir(exist_ok=True)
    root = get(f"{API}/cat/{ROOT_CAT}?poems=false&mainSections=false")["cat"]
    stories = root["children"]
    lines: list[str] = []
    sections = []
    for si, story in enumerate(stories):
        cat = get(f"{API}/cat/{story['id']}?poems=true&mainSections=false")["cat"]
        print(f"[{si + 1}/{len(stories)}] {story['title']}: {len(cat['poems'])} sections")
        for poem in cat["poems"]:
            verses = get(f"{API}/poem/{poem['id']}?{POEM_QUERY}")["verses"]
            couplets: list[str] = []
            half: str | None = None
            for v in sorted(verses, key=lambda v: v["vOrder"]):
                t = clean(v["text"])
                if v["versePosition"] == 0:  # right-hand hemistich
                    half = t
                elif v["versePosition"] == 1 and half is not None:  # left-hand
                    couplets.append(f"{half} / {t}")
                    half = None
                # other positions (headings, prose) do not occur in the epic; skip
            if not couplets:
                continue
            start = len(lines)
            lines.extend(couplets)
            sections.append(
                {
                    "ganjoor_id": poem["id"],
                    "story": story["title"],
                    "story_slug": story["fullUrl"],
                    "title": poem["title"],
                    "first_line": start,
                    "n_couplets": len(couplets),
                }
            )
            lines.append("")  # blank line between sections
            time.sleep(0.05)
    (OUT / "shahnameh.txt").write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    (OUT / "sections.json").write_text(json.dumps(sections, ensure_ascii=False, indent=1), encoding="utf-8")
    n = sum(s["n_couplets"] for s in sections)
    print(f"wrote {n} couplets in {len(sections)} sections")


if __name__ == "__main__":
    main()
