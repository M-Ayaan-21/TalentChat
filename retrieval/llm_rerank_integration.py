import os
from typing import List, Dict, Any
from .llm_reranker import rerank_resumes_with_openai

def build_snippet(candidate: Dict[str, Any], max_chars: int = 1200, max_chunks: int = 2) -> str:
    """
    candidate expected keys:
      - chunks: List[{text: str, score: float, ...}]
      - summary/raw_text (optional)
    """
    chunks = sorted((candidate.get("chunks") or []),
                    key=lambda c: float(c.get("score", 0)), reverse=True)[:max_chunks]
    parts: List[str] = []
    if chunks:
        per = max_chars // max(1, len(chunks))
        for ch in chunks:
            t = (ch.get("text") or "")[:per]
            if t:
                parts.append(t)
    snippet = " ".join(parts).strip()
    if not snippet:
        # fallback if no chunks or empty
        snippet = (candidate.get("summary") or candidate.get("raw_text") or "")[:max_chars]
    return snippet

def apply_llm_rerank(jd_text: str, initial_candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    initial_candidates (input) shape per item:
    {
      "resume_id": "string",                 # REQUIRED (unique stable id for resume)
      "candidate_name": "string",            # optional
      "score": 0.0,                          # your pre-rank score (kept as fallback)
      "chunks": [                            # recommended: top chunks contributing to score
        {"text": "str", "score": 0.0}, ...
      ],
      "skills": ["java","kafka"],            # optional, boosts LLM context
      "location": "Bengaluru",               # optional
      "years_experience": 3,                 # optional
      ... (any other fields you already return)
    }
    Returns the same candidate objects re-ordered, with 'llm_rank' and 'llm_reason' added.
    """
    if not initial_candidates:
        return []

    pool_for_llm: List[Dict[str, Any]] = []
    for c in initial_candidates[:25]:  # pool cap before LLM
        pool_for_llm.append({
            "resume_id": c.get("resume_id") or c.get("id"),
            "name": c.get("candidate_name"),
            "snippet": build_snippet(c),
            "skills": c.get("skills", []),
            "location": c.get("location"),
            "years_experience": c.get("years_experience"),
            "__ref": c,  # keep back-reference to original object
        })

    res = rerank_resumes_with_openai(
        jd_text,
        pool_for_llm,
        top_k=top_k,
        min_score=int(os.getenv("LLM_RERANK_MIN_SCORE", "55"))
    )

    if not res.get("ok") or not res.get("ranked"):
        # LLM failure or nothing above threshold: return your current ordering (top_k)
        return initial_candidates[:top_k]

    id_to_candidate = {str(x["resume_id"]): x["__ref"] for x in pool_for_llm}
    reranked: List[Dict[str, Any]] = []
    for item in res["ranked"]:
        rid = str(item.get("resume_id"))
        base = id_to_candidate.get(rid)
        if not base:
            continue
        out = dict(base)
        out["llm_rank"] = item.get("score")
        out["llm_reason"] = item.get("reason")
        reranked.append(out)

    # Optional: top up with original ranking if LLM returned fewer than requested
    if len(reranked) < top_k:
        seen = {r["resume_id"] for r in reranked}
        for c in initial_candidates:
            rid = c.get("resume_id") or c.get("id")
            if rid not in seen:
                reranked.append(c)
                if len(reranked) >= top_k:
                    break

    return reranked
