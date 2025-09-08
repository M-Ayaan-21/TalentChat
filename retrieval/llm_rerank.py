import json
import os
import re
from typing import List, Dict, Any

from openai import OpenAI
from django_core.config import Config


def _build_prompt(job_description: str, candidates: List[Dict[str, Any]], top_k: int) -> str:
    lines = []
    lines.append("You are an expert technical recruiter. Rank the most suitable candidates for the job.")
    lines.append("")
    lines.append("Job Description:")
    lines.append(job_description.strip())
    lines.append("")
    lines.append("Candidates:")
    for i, c in enumerate(candidates, 1):
        summary_parts = []
        for ch in c.get("matched_chunks", [])[:3]:
            snippet = (ch.get("content") or "").replace("\n", " ").strip()
            if len(snippet) > 250:
                snippet = snippet[:250] + "..."
            summary_parts.append(snippet)
        summary = " ".join(summary_parts) or "[No content]"
        lines.append(f"{i}. ID: {c['candidate_id']} | Name: {c.get('candidate_name','Unknown')} | Score: {c.get('score',0):.2f}")
        lines.append(f"   Summary: {summary}")
        lines.append("")
    lines.append(
        f"""Return ONLY valid JSON (no markdown fences), exactly:
[
  {{"candidate_id":"<existing id>", "rank":1, "reason":"Brief objective reason"}},
  ...
]

Rules:
- Provide up to top {top_k} entries (fewer if fewer candidates).
- No trailing comma after the last object.
- rank starts at 1, increasing by 1; no duplicates.
- Fields allowed: candidate_id, rank, reason.
- Do NOT include explanatory text outside the JSON array."""
    )
    return "\n".join(lines)


def _clean_llm_output(raw: str) -> str:
    if not raw:
        return ""
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = raw.replace("```", "")
    raw = raw.lstrip("\ufeff").strip()
    # isolate first JSON-like array
    match = re.search(r"\[\s*{.*}\s*\]", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    raw = re.sub(r",\s*]", "]", raw)  # remove trailing commas
    raw = re.sub(r",\s*,", ",", raw)
    return raw.strip()


def _parse_json(raw: str) -> List[Dict[str, Any]]:
    cleaned = _clean_llm_output(raw)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        # second attempt extract array
        match = re.search(r"\[\s*{.*}\s*\]", cleaned, re.DOTALL)
        if match:
            attempt = re.sub(r",\s*]", "]", match.group(0))
            try:
                parsed2 = json.loads(attempt)
                if isinstance(parsed2, list):
                    return parsed2
            except Exception:
                pass
    raise ValueError("Unable to parse LLM ranking JSON.")


def rerank_candidates_with_llm(
    job_description: str,
    candidates: List[Dict[str, Any]],
    model: str = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    api_key = Config.OPEN_AI_KEY or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    client = OpenAI(api_key=api_key)
    use_model = model or Config.GPT_3_MODEL or "gpt-3.5-turbo"

    prompt = _build_prompt(job_description, candidates[:20], top_k)

    resp = client.chat.completions.create(
        model=use_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.15,
        max_tokens=700,
    )

    raw = resp.choices[0].message.content or ""

    try:
        parsed = _parse_json(raw)
    except Exception as e:
        # Log and fallback (caller already has ordering)
        import logging
        logging.getLogger(__name__).warning("LLM parsing failed; using original ordering: %s", e)
        fallback = []
        for i, c in enumerate(candidates[:top_k], 1):
            cp = c.copy()
            cp["llm_rank"] = i
            cp["llm_reason"] = ""
            fallback.append(cp)
        return fallback

    id_map = {str(c["candidate_id"]): c for c in candidates}
    ranked: List[Dict[str, Any]] = []

    seen_ranks = set()
    for obj in parsed:
        if not isinstance(obj, dict):
            continue
        cid = str(obj.get("candidate_id"))
        if cid not in id_map:
            continue
        try:
            rnk = int(obj.get("rank"))
        except Exception:
            continue
        if rnk in seen_ranks or rnk < 1:
            continue
        seen_ranks.add(rnk)
        reason = obj.get("reason", "")
        base = id_map[cid].copy()
        base["llm_rank"] = rnk
        base["llm_reason"] = reason
        ranked.append(base)

    # Sort by llm_rank
    ranked.sort(key=lambda x: x.get("llm_rank", 999))

    # Fill if fewer than top_k
    if len(ranked) < top_k:
        have_ids = {r["candidate_id"] for r in ranked}
        for c in candidates:
            if c["candidate_id"] not in have_ids:
                cp = c.copy()
                cp["llm_rank"] = 999
                cp["llm_reason"] = ""
                ranked.append(cp)
                if len(ranked) >= top_k:
                    break
        ranked.sort(key=lambda x: x.get("llm_rank", 999))
        for i, c in enumerate(ranked[:top_k], 1):
            c["llm_rank"] = i

    return ranked[:top_k]
