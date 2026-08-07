#!/usr/bin/env python3
"""Convert copied Kaggle discussion HTML/text to searchable markdown."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "discussion"


class MarkdownHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.href_stack: list[str | None] = []
        self.in_pre = False
        self.in_code = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag in {"p", "div", "section", "article", "li"}:
            self.parts.append("\n")
        elif tag in {"br"}:
            self.parts.append("\n")
        elif tag in {"pre"}:
            self.in_pre = True
            self.parts.append("\n```text\n")
        elif tag in {"code"} and not self.in_pre:
            self.in_code = True
            self.parts.append("`")
        elif tag == "a":
            self.href_stack.append(attrs_dict.get("href"))
            self.parts.append("[")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "div", "section", "article", "li"}:
            self.parts.append("\n")
        elif tag == "pre":
            self.in_pre = False
            self.parts.append("\n```\n")
        elif tag == "code" and self.in_code:
            self.in_code = False
            self.parts.append("`")
        elif tag == "a":
            href = self.href_stack.pop() if self.href_stack else None
            self.parts.append(f"]({href})" if href else "]")

    def handle_data(self, data: str) -> None:
        if self.in_pre:
            self.parts.append(data)
        else:
            self.parts.append(re.sub(r"\s+", " ", data))

    def markdown(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        return text.strip()


def convert(raw: str) -> str:
    if "<" not in raw or ">" not in raw:
        return raw.strip()
    parser = MarkdownHTMLParser()
    parser.feed(raw)
    return parser.markdown()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", help="Input HTML/text file; stdin if omitted")
    parser.add_argument("--title", default=None)
    parser.add_argument("--url", default=None)
    parser.add_argument("--output-dir", default="docs/discussions")
    parser.add_argument("--slug", default=None)
    args = parser.parse_args()

    raw = Path(args.input).read_text(encoding="utf-8") if args.input else sys.stdin.read()
    title = args.title or "Kaggle Discussion"
    slug_source = args.slug or args.title or ""
    if not slug_source and args.url:
        slug_source = Path(urlparse(args.url).path).name
    slug = slugify(slug_source or title)

    body = convert(raw)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slug}.md"
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    front = [f"# {title}", "", f"- archived_at: {now}"]
    if args.url:
        front.append(f"- source: {args.url}")
    front.extend(["", body, ""])
    out_path.write_text("\n".join(front), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
