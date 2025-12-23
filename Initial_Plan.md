# Refined Plan v1.1 (Strategy-Card RAG + RLVER-style training)

## Goal
Train an empathetic dialogue agent that produces *evidence-grounded* psychological-support responses by:
1) selecting an appropriate intervention strategy,
2) retrieving relevant strategy cards,
3) generating a response that follows card steps and uses safe, non-clinical language.

Non-goal: clinical diagnosis or medical treatment advice.

## Key design choice: decision trace (NOT full chain-of-thought)
Instead of requiring long `<think>` reasoning, the policy emits a short, auditable "decision trace" + the user-facing response.

### Output contract (strict)
- `<trace>{...JSON...}</trace>`
- `<answer>...</answer>`

`trace` schema:
- `risk_level`: low/med/high (+ `crisis_flag`: true/false)
- `user_need`: empathize | clarify | coping_skill | action_plan | referral
- `strategy_family`: CBT | MI | DBT | ACT | Supportive | PFA | PM+
- `strategy_id`: string
- `retrieve`: yes/no
- `query`: string (if retrieve=yes)
- `used_card_ids`: list[string] (must be non-empty if retrieve=yes)

## Knowledge base (strategy cards)
Source priority: reputable manuals/guidelines (university / hospital / government / professional org).
Avoid random blogs.

Each card as JSONL:
- `card_id`, `title`
- `when_to_use` (conditions)
- `when_not_to_use` (contraindications / risk gating)
- `goal`
- `steps` (ordered, actionable, minimal)
- `micro_dialogues` (2–4 mini examples)
- `tone_variants` (short / warm / action-oriented)
- `metadata`: problem_type, risk_level, family, keywords, source, section/page

## Retrieval
Hybrid retrieval:
- BM25 + dense embeddings
Return `top_k` cards with versioned index.
For evaluation: deterministic retrieval (fixed index version).
For training: optionally add small retrieval noise (to reduce overfitting).

## Simulator (RLVER-style)
LLM-based affective user simulator:
`step(history, assistant_answer, profile) -> (user_reply, emotion_score, done, info)`

Requirements:
- deterministic emotion score given same inputs
- diverse personas/goals to reduce reward hacking
- versioned prompts + fixed temperature

## Training stages (recommended)
Stage 0: build KB + retrieval + deterministic evaluation harness.
Stage 1 (SFT): supervised fine-tune on small curated set to learn:
  - output format
  - safe style (no diagnosis, no guarantees)
  - grounding to cards (cite steps)
Stage 2 (RL, GRPO first):
  - reward = terminal emotion score (or delta; run ablation)
  - add auxiliary rewards:
    - format compliance
    - grounding/faithfulness: actionable advice must map to retrieved steps
    - retrieval cost: discourage unnecessary retrieval
    - safety: high-risk without crisis/referral => strong penalty
    - overconfidence penalty: "diagnosis / certainty / medical claims"

## Evaluation
Core metrics:
- terminal emotion improvement
- retrieval rate
- grounding/faithfulness rate on actionable steps
- safety compliance rate (risk-triggered behaviors)

Ablations:
1) no retrieval (parametric only)
2) retrieval without grounding reward
3) + grounding/faithfulness reward
4) + retrieval cost
5) GRPO vs PPO
