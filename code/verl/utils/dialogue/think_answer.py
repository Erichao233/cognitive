import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ThinkAnswerParse:
    think: Optional[str]
    answer: Optional[str]
    ok: bool


_TAG_RE_CACHE: dict[str, re.Pattern] = {}


def _tag_re(tag: str) -> re.Pattern:
    if tag not in _TAG_RE_CACHE:
        _TAG_RE_CACHE[tag] = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL | re.IGNORECASE)
    return _TAG_RE_CACHE[tag]


def extract_tag(text: str, tag: str) -> Optional[str]:
    if not text:
        return None
    m = _tag_re(tag).search(text)
    if not m:
        return None
    content = m.group(1).strip()
    return content or None


def parse_think_answer(text: str) -> ThinkAnswerParse:
    think = extract_tag(text, "think")
    answer = extract_tag(text, "answer")
    ok = bool(think is not None and answer is not None)
    return ThinkAnswerParse(think=think, answer=answer, ok=ok)


def extract_query_from_think(think: Optional[str]) -> Optional[str]:
    if not think:
        return None
    for line in think.splitlines():
        s = line.strip()
        if not s:
            continue
        lower = s.lower()
        if lower.startswith("query:"):
            return s.split(":", 1)[1].strip() or None
        if s.startswith("查询:") or s.startswith("检索:"):
            return s.split(":", 1)[1].strip() or None
    return None


def extract_summary_from_think(think: Optional[str]) -> Optional[str]:
    if not think:
        return None
    for line in think.splitlines():
        s = line.strip()
        if not s:
            continue
        lower = s.lower()
        if lower.startswith("summary:"):
            return s.split(":", 1)[1].strip() or None
        if s.startswith("总结:") or s.startswith("摘要:"):
            return s.split(":", 1)[1].strip() or None
    return None


def ensure_closed_tags(think: str, answer: str) -> str:
    return f"<think>\n{think.strip()}\n</think>\n<answer>\n{answer.strip()}\n</answer>"
