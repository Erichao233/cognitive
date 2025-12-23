import hashlib
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import requests


def _sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _canonical_card_id(card: Dict[str, Any]) -> str:
    blob = json.dumps(card, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return _sha1_bytes(blob)


def _card_to_text(card: Dict[str, Any]) -> str:
    parts = [
        f"title: {card.get('title','')}",
        f"conditions: {', '.join(card.get('conditions', []))}",
        f"goal: {card.get('goal','')}",
        f"steps: {' | '.join(card.get('steps', []))}",
        f"contraindications: {' | '.join(card.get('contraindications', []))}",
        f"example_phrases: {' | '.join(card.get('example_phrases', []))}",
    ]
    metadata = card.get("metadata", {})
    if isinstance(metadata, dict):
        keywords = metadata.get("keywords", [])
        if keywords:
            parts.append(f"keywords: {', '.join(keywords)}")
        family = metadata.get("strategy_family", "")
        if family:
            parts.append(f"strategy_family: {family}")
    return "\n".join([p for p in parts if p])


def _post_json(url: str, payload: Dict[str, Any], api_key: str, timeout: int = 300) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _embed_texts(texts: List[str], api_key: str, base_url: str, model: str, batch_size: int = 64) -> np.ndarray:
    url = f"{base_url}/embeddings"
    embeddings = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        payload = {"model": model, "input": chunk}
        data = _post_json(url, payload, api_key)
        for item in data["data"]:
            embeddings.append(item["embedding"])
    return np.array(embeddings, dtype=np.float32)


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return mat / norms


@dataclass
class RetrievedCard:
    uid: str
    chunk_id: str
    score: float
    card: Dict[str, Any]


class StrategyCardIndex:
    def __init__(self, path: str, top_k: int = 3, max_chars: int = 2000):
        self.path = path
        self.top_k = top_k
        self.max_chars = max_chars

        self.api_key = os.getenv("SILICONFLOW_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("Missing SILICONFLOW_API_KEY for embeddings.")
        self.base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        self.model = os.getenv("SILICONFLOW_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")

        self.cache_dir = os.getenv("RAG_CACHE_DIR", "./rag_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cards = []
        self.card_uids = []
        self.card_texts = []
        self.embeddings = None

        self._load_cards()
        self._load_or_build_embeddings()

    def _load_cards(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                card = json.loads(line)
                uid = _canonical_card_id(card)
                self.cards.append(card)
                self.card_uids.append(uid)
                self.card_texts.append(_card_to_text(card))

    def _cache_key(self) -> str:
        with open(self.path, "rb") as f:
            content_hash = _sha1_bytes(f.read())
        key = f"{content_hash}_{self.model}"
        return _sha1_bytes(key.encode("utf-8"))

    def _load_or_build_embeddings(self) -> None:
        key = self._cache_key()
        emb_path = os.path.join(self.cache_dir, f"{key}.npy")
        meta_path = os.path.join(self.cache_dir, f"{key}.json")

        if os.path.exists(emb_path) and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("card_uids") == self.card_uids:
                self.embeddings = np.load(emb_path)
                self.embeddings = _normalize(self.embeddings)
                return

        embeddings = _embed_texts(self.card_texts, self.api_key, self.base_url, self.model)
        embeddings = _normalize(embeddings)
        np.save(emb_path, embeddings)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"card_uids": self.card_uids}, f, ensure_ascii=False)
        self.embeddings = embeddings

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedCard]:
        if not query:
            return []
        top_k = top_k or self.top_k
        q_emb = _embed_texts([query], self.api_key, self.base_url, self.model)
        q_emb = _normalize(q_emb)[0]
        scores = self.embeddings @ q_emb
        idxs = np.argsort(-scores)[:top_k]
        results = []
        for idx in idxs:
            card = self.cards[idx]
            results.append(
                RetrievedCard(
                    uid=self.card_uids[idx],
                    chunk_id=str(card.get("chunk_id", "")),
                    score=float(scores[idx]),
                    card=card,
                )
            )
        return results

    def retrieve_batch(self, queries: List[str], top_k: Optional[int] = None) -> List[List[RetrievedCard]]:
        top_k = top_k or self.top_k
        if not queries:
            return []
        q_embs = _embed_texts(queries, self.api_key, self.base_url, self.model)
        q_embs = _normalize(q_embs)
        scores = q_embs @ self.embeddings.T
        results = []
        for row in scores:
            idxs = np.argsort(-row)[:top_k]
            row_results = []
            for idx in idxs:
                card = self.cards[idx]
                row_results.append(
                    RetrievedCard(
                        uid=self.card_uids[idx],
                        chunk_id=str(card.get("chunk_id", "")),
                        score=float(row[idx]),
                        card=card,
                    )
                )
            results.append(row_results)
        return results

    def format_cards(self, cards: List[RetrievedCard]) -> str:
        lines = ["Relevant strategy cards:"]
        for item in cards:
            card = item.card
            lines.append(f"- card_uid: {item.uid}")
            lines.append(f"  chunk_id: {card.get('chunk_id','')}")
            lines.append(f"  title: {card.get('title','')}")
            lines.append(f"  goal: {card.get('goal','')}")
            steps = card.get("steps", [])
            if steps:
                lines.append("  steps:")
                for step in steps:
                    lines.append(f"    - {step}")
            contraindications = card.get("contraindications", [])
            if contraindications:
                lines.append("  contraindications:")
                for c in contraindications:
                    lines.append(f"    - {c}")
            examples = card.get("example_phrases", [])
            if examples:
                lines.append("  example_phrases:")
                for ex in examples:
                    lines.append(f"    - {ex}")
        text = "\n".join(lines)
        if len(text) > self.max_chars:
            return text[: self.max_chars] + "\n[truncated]"
        return text
