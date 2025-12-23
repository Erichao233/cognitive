#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust PDFs -> Strategy Cards JSONL via LMDeploy OpenAI-compatible API.

Key features:
- Traverse all PDFs in a directory, write ONE big JSONL file.
- Save raw model outputs for each chunk (debug).
- Parse JSON with multiple fallbacks:
  1) direct json.loads
  2) extract {...} block then json.loads
  3) json-repair (optional)
  4) ask the model to "fix the JSON" and re-parse
- Prompt forces single-line strings to avoid invalid JSON (raw newlines in strings).

Requirements:
  pip install pymupdf requests tqdm jsonschema json-repair

Run:
  python3 pdfs_to_strategy_cards_robust.py \
    --pdf-dir /root/autodl-tmp/PDFs \
    --out /root/autodl-tmp/strategy_cards.jsonl \
    --debug-dir /root/autodl-tmp/card_debug \
    --api-base http://127.0.0.1:23333/v1 \
    --model Qwen/Qwen3-8B

Debug fast:
  python3 pdfs_to_strategy_cards_robust.py --limit-pdfs 1 --limit-chunks 2
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import requests
from tqdm import tqdm
from jsonschema import Draft202012Validator

# optional json repair
try:
    from json_repair import repair_json  # type: ignore
    HAVE_JSON_REPAIR = True
except Exception:
    HAVE_JSON_REPAIR = False


# ----------------------------
# Schema: align to your Initial_Plan.md
# ----------------------------
CARD_SCHEMA = {
    "type": "object",
    "required": [
        "chunk_id",
        "title",
        "conditions",
        "goal",
        "steps",
        "contraindications",
        "example_phrases",
        "metadata",
    ],
    "properties": {
        "chunk_id": {"type": "string"},
        "title": {"type": "string"},
        "conditions": {"type": "array", "items": {"type": "string"}},
        "goal": {"type": "string"},
        "steps": {"type": "array", "items": {"type": "string"}, "minItems": 2},
        "contraindications": {"type": "array", "items": {"type": "string"}},
        "example_phrases": {"type": "array", "items": {"type": "string"}},
        "metadata": {"type": "object"},
    },
}

OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["cards"],
    "properties": {
        "cards": {"type": "array", "items": CARD_SCHEMA, "minItems": 1}
    },
}

_OUTPUT_VALIDATOR = Draft202012Validator(OUTPUT_SCHEMA)


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Unit:
    page: int
    text: str
    is_heading: bool = False


@dataclass
class Chunk:
    page_start: int
    page_end: int
    text: str
    section_hint: Optional[str] = None


# ----------------------------
# PDF extraction
# ----------------------------
def extract_pdf_units(pdf_path: str) -> List[Unit]:
    doc = fitz.open(pdf_path)
    units: List[Unit] = []

    heading_patterns = [
        re.compile(r"^\s*(\d+(\.\d+){0,4})\s+.+$"),
        re.compile(r"^\s*(CHAPTER|Chapter)\s+\d+\b.*$"),
        re.compile(r"^\s*(SECTION|Section)\s+\d+\b.*$"),
        re.compile(r"^\s*(SKILL|Skill)\s*:?.+$"),
        re.compile(r"^\s*(EXERCISE|Exercise)\s*:?.+$"),
        re.compile(r"^\s*(WORKSHEET|Worksheet)\s*:?.+$"),
    ]

    def is_heading_line(line: str) -> bool:
        line = line.strip()
        if not line or len(line) > 120:
            return False
        if 4 <= len(line) <= 80 and line.isupper():
            return True
        return any(p.match(line) for p in heading_patterns)

    for i in range(len(doc)):
        page_no = i + 1
        raw = doc[i].get_text("text") or ""
        raw = raw.replace("\r", "\n")
        raw = re.sub(r"\n{3,}", "\n\n", raw).strip()
        if not raw:
            continue

        blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
        for b in blocks:
            b = re.sub(r"[ \t]{2,}", " ", b).strip()
            first_line = b.split("\n", 1)[0].strip()
            units.append(Unit(page=page_no, text=b, is_heading=is_heading_line(first_line)))

    return units


def make_chunks(units: List[Unit], max_chars: int = 8000, min_chars: int = 1200) -> List[Chunk]:
    chunks: List[Chunk] = []
    cur_texts: List[str] = []
    cur_pages: List[int] = []
    cur_section: Optional[str] = None
    cur_len = 0

    def flush():
        nonlocal cur_texts, cur_pages, cur_section, cur_len
        if not cur_texts:
            return
        txt = "\n\n".join(cur_texts).strip()
        if txt:
            chunks.append(
                Chunk(
                    page_start=min(cur_pages),
                    page_end=max(cur_pages),
                    text=txt,
                    section_hint=cur_section,
                )
            )
        cur_texts, cur_pages, cur_section, cur_len = [], [], None, 0

    for u in units:
        u_len = len(u.text)

        # start a new chunk when a heading appears and current chunk is "large enough"
        if u.is_heading and cur_len >= min_chars:
            flush()
            cur_section = u.text.split("\n", 1)[0].strip()

        # enforce max chunk size
        if cur_texts and (cur_len + u_len > max_chars):
            flush()
            if u.is_heading:
                cur_section = u.text.split("\n", 1)[0].strip()

        cur_texts.append(u.text)
        cur_pages.append(u.page)
        cur_len += u_len

    flush()
    return chunks


# ----------------------------
# LMDeploy OpenAI-compatible calls
# ----------------------------
def post_json(url: str, payload: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def chat_completion(api_base: str,
                    model: str,
                    messages: List[Dict[str, str]],
                    temperature: float,
                    max_tokens: int,
                    timeout: int = 300,
                    retries: int = 3) -> str:
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            data = post_json(url, payload, timeout=timeout)
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            time.sleep(1.2 * attempt)
    raise RuntimeError(f"LMDeploy request failed: {last_err}")


# ----------------------------
# JSON parsing utilities
# ----------------------------
def extract_json_block(text: str) -> str:
    """
    Extract a JSON object substring if model adds extra text.
    """
    text = text.strip()
    # try direct if starts with '{'
    if text.startswith("{") and text.endswith("}"):
        return text
    # fallback: first {...} block
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return m.group(0)


def parse_output_to_obj(raw: str) -> Dict[str, Any]:
    """
    Parse raw model output into Python object. Multiple fallbacks.
    """
    block = extract_json_block(raw)
    # 1) strict
    try:
        return json.loads(block)
    except Exception:
        pass

    # 2) json-repair if available
    if HAVE_JSON_REPAIR:
        try:
            repaired = repair_json(block, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
        except Exception:
            pass

    # 3) last-resort: a minimal cleanup (remove trailing commas)
    # NOTE: not perfect, but catches some cases.
    cleaned = re.sub(r",\s*([\]}])", r"\1", block)
    return json.loads(cleaned)


def validate_cards_obj(obj: Dict[str, Any]) -> None:
    _OUTPUT_VALIDATOR.validate(obj)


def normalize_cards(obj: Dict[str, Any],
                    pdf_name: str,
                    page_start: int,
                    page_end: int,
                    chunk_idx: int,
                    section_hint: Optional[str]) -> List[Dict[str, Any]]:
    validate_cards_obj(obj)
    cards = obj["cards"]
    out: List[Dict[str, Any]] = []
    base_chunk_id = f"{pdf_name}::p{page_start}-{page_end}::chunk{chunk_idx}"

    for k, c in enumerate(cards):
        c = dict(c)
        if not c.get("chunk_id"):
            c["chunk_id"] = f"{base_chunk_id}::card{k}"

        # force single-line strings (avoid JSON re-serialization surprises)
        def one_line(s: str) -> str:
            s = (s or "").replace("\r", " ").replace("\n", " ")
            s = re.sub(r"\s{2,}", " ", s).strip()
            return s

        c["title"] = one_line(c.get("title", ""))
        c["goal"] = one_line(c.get("goal", ""))
        c["conditions"] = [one_line(x) for x in c.get("conditions", []) if one_line(x)]
        c["contraindications"] = [one_line(x) for x in c.get("contraindications", []) if one_line(x)]
        c["steps"] = [one_line(x) for x in c.get("steps", []) if one_line(x)]
        c["example_phrases"] = [one_line(x) for x in c.get("example_phrases", []) if one_line(x)]

        # enforce minimum steps
        if len(c["steps"]) < 2:
            continue

        meta = dict(c.get("metadata", {}))
        meta.update({
            "source_pdf": pdf_name,
            "source_pages": [page_start, page_end],
            "section_hint": section_hint or "",
        })
        c["metadata"] = meta

        out.append(c)

    return out


# ----------------------------
# Prompt builders
# ----------------------------
def build_extract_messages(pdf_name: str,
                           page_start: int,
                           page_end: int,
                           section_hint: Optional[str],
                           chunk_text: str) -> List[Dict[str, str]]:
    system = {
        "role": "system",
        "content": (
            "You are a STRICT JSON generator.\n"
            "Return ONLY one JSON object. No markdown, no explanations.\n"
            "All strings must be SINGLE-LINE (no raw newline characters inside strings).\n"
            "Use DOUBLE QUOTES for JSON strings.\n\n"
            "Schema:\n"
            "{\n"
            "  \"cards\": [\n"
            "    {\n"
            "      \"chunk_id\": \"string\",\n"
            "      \"title\": \"string\",\n"
            "      \"conditions\": [\"string\"],\n"
            "      \"goal\": \"string\",\n"
            "      \"steps\": [\"string\", \"string\"],\n"
            "      \"contraindications\": [\"string\"],\n"
            "      \"example_phrases\": [\"string\"],\n"
            "      \"metadata\": {\"strategy_family\": \"string\", \"keywords\": [\"string\"]}\n"
            "    }\n"
            "  ]\n"
            "}\n"
        )
    }
    hint = f"Section hint: {section_hint}\n" if section_hint else ""
    user = {
        "role": "user",
        "content": (
            f"Source PDF: {pdf_name}\nPages: {page_start}-{page_end}\n{hint}\n"
            "Extract 1-3 strategy cards from the excerpt.\n"
            "Rules:\n"
            "- Steps must be actionable and specific.\n"
            "- Contraindications must be present (when NOT to use / risk caveats).\n"
            "- Keep wording non-clinical: do NOT diagnose, do NOT recommend medication.\n"
            "- example_phrases should be grounded in steps.\n\n"
            "EXCERPT:\n-----\n"
            f"{chunk_text}\n"
            "-----\n"
        )
    }
    return [system, user]


def build_fix_messages(bad_output: str, err_msg: str) -> List[Dict[str, str]]:
    system = {
        "role": "system",
        "content": (
            "You are a JSON repair tool.\n"
            "Fix the input into a SINGLE valid JSON object that matches the exact schema.\n"
            "Return ONLY the corrected JSON object. No markdown, no commentary.\n"
            "All strings must be SINGLE-LINE. Use DOUBLE QUOTES.\n"
        )
    }
    user = {
        "role": "user",
        "content": (
            f"The previous output is invalid JSON.\n"
            f"Parser error: {err_msg}\n"
            "Fix it and return ONLY valid JSON:\n"
            "-----\n"
            f"{bad_output}\n"
            "-----\n"
        )
    }
    return [system, user]


# ----------------------------
# Debug save
# ----------------------------
def save_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", type=str, default="/root/autodl-tmp/PDFs")
    ap.add_argument("--out", type=str, default="/root/autodl-tmp/strategy_cards.jsonl")
    ap.add_argument("--debug-dir", type=str, default="/root/autodl-tmp/card_debug")
    ap.add_argument("--api-base", type=str, default="http://127.0.0.1:23333/v1")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--max-chars", type=int, default=8000)
    ap.add_argument("--min-chars", type=int, default=1200)
    ap.add_argument("--max-tokens", type=int, default=1400)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit-pdfs", type=int, default=0)
    ap.add_argument("--limit-chunks", type=int, default=0)
    ap.add_argument("--skip-empty", action="store_true")
    args = ap.parse_args()

    pdf_paths = sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf")))
    if args.limit_pdfs and args.limit_pdfs > 0:
        pdf_paths = pdf_paths[:args.limit_pdfs]

    if not pdf_paths:
        print(f"[ERROR] No PDFs found in {args.pdf_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(args.debug_dir, exist_ok=True)

    total_cards = 0
    total_fail = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        for pdf_path in tqdm(pdf_paths, desc="PDFs"):
            pdf_name = os.path.basename(pdf_path)

            try:
                units = extract_pdf_units(pdf_path)
            except Exception as e:
                print(f"[WARN] Failed to read {pdf_name}: {e}", file=sys.stderr)
                continue

            if args.skip_empty and len(units) < 5:
                print(f"[WARN] Skip {pdf_name}: extracted text too small.", file=sys.stderr)
                continue

            chunks = make_chunks(units, max_chars=args.max_chars, min_chars=args.min_chars)
            if args.limit_chunks and args.limit_chunks > 0:
                chunks = chunks[:args.limit_chunks]

            for chunk_idx, ch in enumerate(tqdm(chunks, desc=f"Chunks({pdf_name})", leave=False)):
                if len(ch.text.strip()) < 400:
                    continue

                # call model (extract)
                messages = build_extract_messages(
                    pdf_name=pdf_name,
                    page_start=ch.page_start,
                    page_end=ch.page_end,
                    section_hint=ch.section_hint,
                    chunk_text=ch.text
                )

                raw_out = ""
                try:
                    raw_out = chat_completion(
                        api_base=args.api_base,
                        model=args.model,
                        messages=messages,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        retries=3
                    )

                    # Save raw output for every chunk (so you can reprocess later)
                    raw_path = os.path.join(
                        args.debug_dir,
                        f"{pdf_name}.p{ch.page_start}-{ch.page_end}.chunk{chunk_idx}.raw.txt"
                    )
                    save_text(raw_path, raw_out)

                    # parse
                    obj = parse_output_to_obj(raw_out)
                    cards = normalize_cards(
                        obj=obj,
                        pdf_name=pdf_name,
                        page_start=ch.page_start,
                        page_end=ch.page_end,
                        chunk_idx=chunk_idx,
                        section_hint=ch.section_hint
                    )

                except Exception as e1:
                    # 1st failure: try model-based JSON fix
                    total_fail += 1
                    err_msg = str(e1)
                    fix_messages = build_fix_messages(raw_out if raw_out else "(empty)", err_msg)
                    try:
                        fixed = chat_completion(
                            api_base=args.api_base,
                            model=args.model,
                            messages=fix_messages,
                            temperature=0.0,
                            max_tokens=min(1200, args.max_tokens),
                            retries=2
                        )
                        fix_path = os.path.join(
                            args.debug_dir,
                            f"{pdf_name}.p{ch.page_start}-{ch.page_end}.chunk{chunk_idx}.fixed.txt"
                        )
                        save_text(fix_path, fixed)

                        obj = parse_output_to_obj(fixed)
                        cards = normalize_cards(
                            obj=obj,
                            pdf_name=pdf_name,
                            page_start=ch.page_start,
                            page_end=ch.page_end,
                            chunk_idx=chunk_idx,
                            section_hint=ch.section_hint
                        )
                    except Exception as e2:
                        # still failed -> log and continue
                        print(
                            f"\n[WARN] Chunk failed {pdf_name} p{ch.page_start}-{ch.page_end} "
                            f"(extract err: {e1}; fix err: {e2})",
                            file=sys.stderr
                        )
                        snippet = (ch.text[:600] + "...") if len(ch.text) > 600 else ch.text
                        print(f"[DEBUG] Chunk snippet:\n{snippet}\n", file=sys.stderr)
                        continue

                # write cards
                for c in cards:
                    fout.write(json.dumps(c, ensure_ascii=False) + "\n")
                total_cards += len(cards)

    print(
        f"[DONE] cards={total_cards}, failed_chunks={total_fail}, out={args.out}, debug_dir={args.debug_dir}",
        file=sys.stderr
    )


if __name__ == "__main__":
    main()
