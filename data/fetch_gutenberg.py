"""
Fetch public-domain books from Project Gutenberg (via the gutendex API),
strip the standard PG headers/footers, segment into chapters, and emit
CSV(s) in the same `(no, story)` schema the project already uses.

Run:
    .venv/bin/python -m data.fetch_gutenberg \\
        --topics fantasy,mythology,fairy-tale \\
        --limit  250 \\
        --out    data/books/

The gutendex API requires no key and is rate-limit-tolerant.
"""
import argparse
import csv
import json
import re
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path


GUTENDEX = "https://gutendex.com/books"

# Standard markers Project Gutenberg uses to bracket the actual book text.
_START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE)
_END_RE   = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*",   re.IGNORECASE)

# Chapter splitter — matches headings like "CHAPTER I", "Chapter 12", "Chapter XIV.".
_CHAPTER_RE = re.compile(
    r"^\s*(?:CHAPTER|Chapter|chapter)\s+[IVXLCDM0-9]+\b.*$",
    re.MULTILINE,
)


def _http_get(url: str, retries: int = 5, backoff: float = 2.0, timeout: int = 120) -> bytes:
    last_err: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "llm-paraphraser-trainer/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last_err = e
            time.sleep(backoff ** i)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_err}")


def search_books(topics: list[str], limit: int) -> list[dict]:
    """Page through gutendex, filtering to English text books matching the topics.

    Returns a list of book metadata dicts (with `id`, `title`, and `formats`).
    Stops when `limit` entries are collected or pages run out. A failed page
    request only stops *that topic's* pagination — we move on to the next topic
    rather than dying entirely.
    """
    out: list[dict] = []
    for topic in topics:
        url = f"{GUTENDEX}?topic={urllib.parse.quote(topic)}&languages=en"
        page = 1
        while url and len(out) < limit:
            try:
                data = json.loads(_http_get(url).decode("utf-8"))
            except RuntimeError as e:
                print(f"  gutendex pagination failed at topic={topic} page={page}: {e}",
                      file=sys.stderr)
                break
            for b in data.get("results", []):
                fmts = b.get("formats", {})
                # Prefer the plain-UTF8 text format.
                txt_url = (fmts.get("text/plain; charset=utf-8")
                           or fmts.get("text/plain; charset=us-ascii")
                           or fmts.get("text/plain"))
                if not txt_url:
                    continue
                out.append({
                    "id":     b["id"],
                    "title":  b.get("title", f"PG{b['id']}"),
                    "url":    txt_url,
                    "topic":  topic,
                })
                if len(out) >= limit:
                    break
            url = data.get("next")
            page += 1
    return out


def strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg's wrapper text. Returns the inner book content."""
    m_start = _START_RE.search(text)
    if m_start:
        text = text[m_start.end():]
    m_end = _END_RE.search(text)
    if m_end:
        text = text[:m_end.start()]
    return text.strip()


def segment_chapters(text: str) -> list[str]:
    """Split a stripped book into chapter strings. If no chapter headings detected,
    return the whole text as a single chapter (still useful as one long span)."""
    matches = list(_CHAPTER_RE.finditer(text))
    if not matches:
        return [text.strip()]

    chapters: list[str] = []
    # Everything before the first heading is front-matter — discard.
    for i, m in enumerate(matches):
        start = m.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if len(chunk) >= 500:  # skip near-empty chapter stubs
            chapters.append(chunk)
    return chapters or [text.strip()]


def book_to_csv(book: dict, out_dir: Path) -> tuple[Path, int, int]:
    """Download + clean + segment one book. Writes <id>.csv. Returns (path, chapters, chars)."""
    raw    = _http_get(book["url"]).decode("utf-8", errors="replace")
    inner  = strip_gutenberg(raw)
    chaps  = segment_chapters(inner)

    out_path = out_dir / f"pg_{book['id']}.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["no", "story"])
        for i, ch in enumerate(chaps, start=1):
            w.writerow([i, ch])

    return out_path, len(chaps), sum(len(c) for c in chaps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default="fantasy,mythology,fairy-tale",
                    help="Comma-separated gutendex topic strings.")
    ap.add_argument("--limit",  default=250, type=int,
                    help="Maximum number of books to download across all topics.")
    ap.add_argument("--out",    default="data/books", type=Path,
                    help="Output directory (one CSV per book).")
    ap.add_argument("--manifest", default="data/books/_manifest.jsonl", type=Path,
                    help="JSONL log of downloaded books (id, title, topic, path, chars).")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    print(f"Searching gutendex for topics={topics} limit={args.limit} ...", file=sys.stderr)
    books = search_books(topics, args.limit)
    print(f"Found {len(books)} candidate books.", file=sys.stderr)

    # Already-downloaded ids (so re-runs are incremental).
    seen_ids: set[int] = set()
    if args.manifest.exists():
        for line in args.manifest.open():
            try:
                seen_ids.add(json.loads(line)["id"])
            except Exception:
                pass

    total_chars = 0
    total_chaps = 0
    written     = 0
    with args.manifest.open("a", encoding="utf-8") as mf:
        for i, book in enumerate(books, start=1):
            if book["id"] in seen_ids:
                continue
            try:
                path, n_chaps, n_chars = book_to_csv(book, args.out)
            except Exception as e:
                print(f"  [{i}/{len(books)}] FAIL pg_{book['id']}: {e}", file=sys.stderr)
                continue
            total_chars += n_chars
            total_chaps += n_chaps
            written     += 1
            mf.write(json.dumps({
                "id":     book["id"],
                "title":  book["title"],
                "topic":  book["topic"],
                "path":   str(path),
                "chapters": n_chaps,
                "chars":  n_chars,
            }) + "\n")
            print(f"  [{i}/{len(books)}] pg_{book['id']:>6} chaps={n_chaps:>3} "
                  f"chars={n_chars:>7,}  {book['title'][:60]}", file=sys.stderr)

    print(f"\nDownloaded {written} new books "
          f"({total_chaps} chapters, {total_chars:,} chars).", file=sys.stderr)


if __name__ == "__main__":
    main()
