# MVP Plan (Strategy-Card Vector RAG + RLVER-style training)

## Goal
Train an empathetic dialogue agent that produces *evidence-grounded* psychological-support responses by:
1) retrieving relevant strategy cards,
2) generating responses informed by retrieved steps (no explicit trace format in MVP).

Non-goal: clinical diagnosis or medical treatment advice.

## MVP design choice: no explicit trace format
For MVP, we skip structured `<trace>` outputs and focus on:
- retrieval injection into model context
- terminal emotion score reward

## Knowledge base (strategy cards)
Source priority: reputable manuals/guidelines (university / hospital / government / professional org).
Avoid random blogs.

Current JSONL schema (aligned with `data/strategy_cards.jsonl`):
- `chunk_id` (NOT unique; use a hashed `card_uid` in code)
- `title`
- `conditions` (list)
- `goal`
- `steps` (list)
- `contraindications` (list)
- `example_phrases` (list)
- `metadata` (dict: `strategy_family`, `keywords`, `source_pdf`, `source_pages`, `section_hint`)

## Retrieval (MVP)
Vector retrieval only:
- SiliconFlow embeddings (`Qwen/Qwen3-Embedding-4B`)
- cosine similarity + top_k
- cache embeddings on disk per worker

## Simulator (RLVER-style)
LLM-based affective user simulator:
`step(history, assistant_answer, profile) -> (user_reply, emotion_score, done, info)`

MVP requirements:
- deterministic via `temperature=0` + prompt caching
- fixed prompt template

## Training stages (MVP)
Stage 0: build KB + retrieval + deterministic simulator reward.
Stage 1 (RL only, GRPO):
  - reward = terminal emotion score (or delta; run ablation)
  - no auxiliary rewards for MVP

## Evaluation (MVP)
Core metrics:
- terminal emotion improvement
- retrieval rate

Ablations (later):
1) no retrieval (parametric only)
2) add grounding/faithfulness reward
3) add retrieval cost
