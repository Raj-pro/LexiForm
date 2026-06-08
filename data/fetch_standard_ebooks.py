"""
Fetch public-domain books from Standard Ebooks.

NOTE: As of 2026-06, Standard Ebooks' OPDS feed at /opds/all requires HTTP
Basic Auth (returns 401 for anonymous clients). To use this script you must
register a free account at https://standardebooks.org/contributors/sign-up
and pass --user / --password (or set SE_USER / SE_PASS env vars).

Their catalog is small (~700 books) compared to Gutenberg (~70K), so this is
optional — fetch_gutenberg.py alone provides enough text for Phase 2.

Run (with credentials):
    SE_USER=you@example.com SE_PASS=... \\
    .venv/bin/python -m data.fetch_standard_ebooks \\
        --filter fantasy --limit 30 --out data/books/

The filter is a substring match against title/author/subject (case-insensitive).
Uses stdlib only (urllib + xml.etree + zipfile + html.parser) — no new deps.
"""
import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import time
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET


OPDS_URL = "https://standardebooks.org/opds/all"

# Set once from CLI / env in main(); _http_get reads this for the auth header.
_AUTH_HEADER: str | None = None

# Atom + OPDS namespaces.
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "dc":   "http://purl.org/dc/terms/",
}


def _http_get(url: str, retries: int = 3, backoff: float = 1.5) -> bytes:
    last_err: Exception | None = None
    headers = {"User-Agent": "llm-paraphraser-trainer/1.0"}
    if _AUTH_HEADER is not None:
        headers["Authorization"] = _AUTH_HEADER
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            last_err = e
            time.sleep(backoff ** (i + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_err}")


class _TextExtractor(HTMLParser):
    """Concatenate all visible text in an XHTML document. Drops <script>/<style>."""
    def __init__(self) -> None:
        super().__init__()
        self.buf:  list[str] = []
        self.skip: int = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in {"script", "style"}:
            self.skip += 1
        elif tag in {"p", "br", "h1", "h2", "h3", "h4", "li", "div"}:
            self.buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.skip > 0:
            self.skip -= 1
        elif tag in {"p", "h1", "h2", "h3", "h4", "li", "div"}:
            self.buf.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip == 0:
            self.buf.append(data)

    def text(self) -> str:
        joined = "".join(self.buf)
        return re.sub(r"\n{3,}", "\n\n", joined).strip()


def list_books(filter_substr: str, limit: int) -> list[dict]:
    """Parse the OPDS feed, return entries matching `filter_substr` (case-insensitive)
    in title/author/subject. EPUB download URL is the rel="http://opds-spec.org/acquisition" link.
    """
    xml = _http_get(OPDS_URL)
    root = ET.fromstring(xml)
    needle = filter_substr.lower()
    out: list[dict] = []
    for entry in root.findall("atom:entry", NS):
        title = (entry.findtext("atom:title", default="", namespaces=NS) or "").strip()
        author_el = entry.find("atom:author/atom:name", NS)
        author = (author_el.text or "").strip() if author_el is not None else ""
        subjects = [(c.get("label") or c.get("term") or "")
                    for c in entry.findall("atom:category", NS)]
        haystack = " ".join([title, author, *subjects]).lower()
        if needle and needle not in haystack:
            continue

        epub_url = None
        for link in entry.findall("atom:link", NS):
            if (link.get("rel", "").startswith("http://opds-spec.org/acquisition")
                    and "epub" in (link.get("type", "") or "").lower()):
                epub_url = link.get("href")
                break
        if not epub_url:
            continue
        if epub_url.startswith("/"):
            epub_url = "https://standardebooks.org" + epub_url

        out.append({
            "title":  title,
            "author": author,
            "url":    epub_url,
            "subjects": subjects,
        })
        if len(out) >= limit:
            break
    return out


def epub_to_chapters(epub_bytes: bytes) -> list[str]:
    """Open an EPUB (zipfile of XHTML), extract text from each chapter file,
    return non-empty chapters ordered by spine position when possible.
    """
    chapters: list[tuple[str, str]] = []  # (filename, text)
    with zipfile.ZipFile(io.BytesIO(epub_bytes)) as z:
        for name in z.namelist():
            lname = name.lower()
            if not (lname.endswith(".xhtml") or lname.endswith(".html")):
                continue
            # Skip nav/toc/copyright/colophon pages that aren't story content.
            if any(bad in lname for bad in
                   ("toc", "nav", "colophon", "uncopyright", "imprint",
                    "halftitle", "titlepage", "copyright")):
                continue
            try:
                raw = z.read(name).decode("utf-8", errors="replace")
            except Exception:
                continue
            tp = _TextExtractor()
            tp.feed(raw)
            txt = tp.text()
            if len(txt) >= 500:
                chapters.append((name, txt))
    # Order by filename (Standard Ebooks uses chapter-XX.xhtml).
    chapters.sort(key=lambda x: x[0])
    return [t for _, t in chapters]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter",   default="fantasy",
                    help="Substring filter against title/author/subject "
                         "(case-insensitive).")
    ap.add_argument("--limit",    default=50, type=int)
    ap.add_argument("--out",      default="data/books", type=Path)
    ap.add_argument("--manifest", default="data/books/_se_manifest.jsonl", type=Path)
    ap.add_argument("--user",     default=os.environ.get("SE_USER", ""),
                    help="Standard Ebooks username (or set SE_USER env var).")
    ap.add_argument("--password", default=os.environ.get("SE_PASS", ""),
                    help="Standard Ebooks password (or set SE_PASS env var).")
    args = ap.parse_args()

    if args.user and args.password:
        global _AUTH_HEADER
        token = base64.b64encode(f"{args.user}:{args.password}".encode()).decode()
        _AUTH_HEADER = f"Basic {token}"
    else:
        print("WARNING: no Standard Ebooks credentials set; /opds/all will 401. "
              "Use --user/--password or SE_USER/SE_PASS env vars.", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Listing Standard Ebooks (filter={args.filter!r}, limit={args.limit}) ...",
          file=sys.stderr)
    books = list_books(args.filter, args.limit)
    print(f"Matched {len(books)} books.", file=sys.stderr)

    seen_urls: set[str] = set()
    if args.manifest.exists():
        for line in args.manifest.open():
            try:
                seen_urls.add(json.loads(line)["url"])
            except Exception:
                pass

    total_chars = 0
    total_chaps = 0
    written     = 0
    with args.manifest.open("a", encoding="utf-8") as mf:
        for i, b in enumerate(books, start=1):
            if b["url"] in seen_urls:
                continue
            try:
                epub = _http_get(b["url"])
                chapters = epub_to_chapters(epub)
            except Exception as e:
                print(f"  [{i}/{len(books)}] FAIL {b['title'][:50]}: {e}", file=sys.stderr)
                continue
            if not chapters:
                continue
            slug = re.sub(r"[^a-zA-Z0-9]+", "-", b["title"].lower()).strip("-")[:60]
            out_path = args.out / f"se_{slug}.csv"
            with out_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["no", "story"])
                for j, ch in enumerate(chapters, start=1):
                    w.writerow([j, ch])
            n_chars = sum(len(c) for c in chapters)
            total_chars += n_chars
            total_chaps += len(chapters)
            written     += 1
            mf.write(json.dumps({
                "url":       b["url"],
                "title":     b["title"],
                "author":    b["author"],
                "path":      str(out_path),
                "chapters":  len(chapters),
                "chars":     n_chars,
            }) + "\n")
            print(f"  [{i}/{len(books)}] {b['title'][:50]:<50} "
                  f"chaps={len(chapters):>3} chars={n_chars:>7,}", file=sys.stderr)

    print(f"\nDownloaded {written} new books "
          f"({total_chaps} chapters, {total_chars:,} chars).", file=sys.stderr)


if __name__ == "__main__":
    main()
