from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.llm import LLMClient, build_llm_client
from agentic_rag_eval.logging_setup import get_logger
from agentic_rag_eval.prompts import get_prompt_registry
from agentic_rag_eval.schemas import EvalRecord, LLMMessage, Passage

logger = get_logger(__name__)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON object parser for LLM outputs."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT_RE.search(text)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            return None
    return None


def _format_reasoning(reasoning: Any) -> str:
    if reasoning is None:
        return ""
    if isinstance(reasoning, str):
        return reasoning
    if isinstance(reasoning, list):
        parts: list[str] = []
        for i, step in enumerate(reasoning, start=1):
            if isinstance(step, str):
                parts.append(f"Step {i}: {step}")
            elif isinstance(step, dict):
                thought = step.get("thought", "")
                action = step.get("action", "")
                obs = step.get("observation", "")
                parts.append(f"Step {i}: thought={thought} action={action} obs={obs}")
            else:
                parts.append(f"Step {i}: {step}")
        return "\n".join(parts)
    return str(reasoning)


def _format_evidence(evidence: Iterable[Any]) -> str:
    lines: list[str] = []
    for i, ev in enumerate(evidence, start=1):
        if isinstance(ev, Passage):
            title = ev.title or ev.passage_id
            text = ev.text
        elif isinstance(ev, dict):
            title = ev.get("title") or ev.get("passage_id", f"p{i}")
            text = ev.get("text", "")
        else:
            title = f"p{i}"
            text = str(ev)
        lines.append(f"[{i}] {title}: {text}")
    return "\n".join(lines)


class LLMJudge:
    """Score reasoning coherence and answer completeness via the eval LLM."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        eval_llm: LLMClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._eval_llm = eval_llm or build_llm_client(self._settings, role="eval")
        self._registry = get_prompt_registry()

    def score_coherence(
        self,
        question: str,
        answer: str,
        reasoning: Any,
        evidence: Iterable[Any],
    ) -> dict[str, Any]:
        """Score reasoning coherence on a 1-5 scale; returns ``{score, rationale}``."""
        prompt = self._registry.get("judge_coherence").render(
            question=question,
            answer=answer,
            reasoning=_format_reasoning(reasoning),
            evidence=_format_evidence(evidence),
        )
        return self._call_and_parse(prompt, kind="coherence")

    def score_completeness(
        self,
        question: str,
        gold_answer: str,
        predicted_answer: str,
    ) -> dict[str, Any]:
        """Score answer completeness on a 1-5 scale; returns ``{score, rationale}``."""
        prompt = self._registry.get("judge_completeness").render(
            question=question,
            gold_answer=gold_answer,
            predicted_answer=predicted_answer,
        )
        return self._call_and_parse(prompt, kind="completeness")

    def _call_and_parse(self, prompt: str, *, kind: str) -> dict[str, Any]:
        try:
            result = self._eval_llm.complete(
                [LLMMessage(role="user", content=prompt)],
                temperature=0.0,
            )
        except Exception as e:
            logger.warning("judge_call_failed", extra={"kind": kind, "error": str(e)})
            return {"score": None, "rationale": f"llm_error: {e}"}

        data = _extract_json(result.content)
        if data is None:
            logger.warning(
                "judge_parse_failed",
                extra={"kind": kind, "content": result.content[:500]},
            )
            return {"score": None, "rationale": "parse_error"}

        raw_score = data.get("score")
        score: int | None
        try:
            if raw_score is None:
                score = None
            else:
                s = round(float(raw_score))
                if not 1 <= s <= 5:
                    logger.warning("judge_score_out_of_range", extra={"kind": kind, "score": s})
                    score = None
                else:
                    score = s
        except (TypeError, ValueError):
            score = None

        rationale = str(data.get("rationale", "")).strip()
        return {"score": score, "rationale": rationale}

    def calibrate(self, records: list[EvalRecord]) -> dict[str, Any]:
        """Return Cohen's kappa between binarized judge completeness and HotpotQA F1."""
        pairs: list[tuple[int, int]] = []
        for r in records:
            js = r.judge_completeness
            if js is None:
                continue
            judge_bin = 1 if float(js) >= 4 else 0
            gold_bin = 1 if float(r.f1) >= 0.5 else 0
            pairs.append((judge_bin, gold_bin))

        if not pairs:
            return {"kappa": None, "n": 0, "reason": "no_judge_scores"}

        kappa = _cohens_kappa(pairs)
        counts = {"11": 0, "10": 0, "01": 0, "00": 0}
        for j, g in pairs:
            counts[f"{j}{g}"] += 1
        return {"kappa": kappa, "n": len(pairs), "counts": counts}


def _cohens_kappa(pairs: list[tuple[int, int]]) -> float:
    """Cohen's kappa for binary ratings (returns 0.0 when either rater is constant)."""
    n = len(pairs)
    if n == 0:
        return 0.0

    agree = sum(1 for j, g in pairs if j == g)
    po = agree / n

    j_pos = sum(1 for j, _ in pairs if j == 1) / n
    g_pos = sum(1 for _, g in pairs if g == 1) / n
    pe = j_pos * g_pos + (1 - j_pos) * (1 - g_pos)

    if pe >= 1.0:
        return 0.0
    return (po - pe) / (1.0 - pe)
