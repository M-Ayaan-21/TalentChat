import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = os.getenv("OPENAI_RERANK_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Limits to control latency/cost
MAX_RESUMES = int(os.getenv("RERANK_MAX_RESUMES", "25"))
MAX_SNIPPET_CHARS = int(os.getenv("RERANK_MAX_SNIPPET_CHARS", "1200"))
REQUEST_TIMEOUT = int(os.getenv("RERANK_TIMEOUT_SECS", "25"))

def _hash_key(jd_text: str, candidates: List[Dict[str, Any]]) -> str:
    h = hashlib.sha1()
    h.update(jd_text.encode("utf-8", errors="ignore"))
    for c in candidates:
        rid = str(c.get("resume_id") or c.get("id") or "")
        snippet = (c.get("snippet") or c.get("text") or "")[:MAX_SNIPPET_CHARS]
        h.update(rid.encode("utf-8", errors="ignore"))
        h.update(snippet.encode("utf-8", errors="ignore"))
    return h.hexdigest()

def _build_prompt(jd_text: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    payload = {
        "job_description": jd_text.strip(),
        "candidates": [
            {
                "resume_id": str(c.get("resume_id") or c.get("id")),
                "name": (c.get("name") or c.get("candidate_name") or "")[:80],
                "snippet": (c.get("snippet") or c.get("text") or "")[:MAX_SNIPPET_CHARS],
                "skills": c.get("skills", [])[:30],
                "location": c.get("location", "")[:80],
                "years_experience": c.get("years_experience", "")
            }
            for c in candidates
        ]
    }
    system = (
        "You are a precise technical recruiter. Rank resumes by job fit.\n"
        "Scoring rubric (0-100):\n"
        "- Must-have skills/tools in JD (40%)\n"
        "- Seniority match (20%)\n"
        "- Domain/role alignment (20%)\n"
        "- Recency/relevant experience (10%)\n"
        "- Nice-to-haves/location (10%)\n"
        "Penalize generic-only matches (e.g., just 'developer').\n"
        "Return STRICT JSON as {\"ranked\": [{\"resume_id\": str, \"score\": int, \"reason\": str} ...]} "
        "sorted by descending score. No extra text."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
    ]

def rerank_resumes_with_openai(
    jd_text: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    min_score: int = 55,
    use_cache: bool = True,
    cache: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    if not candidates:
        return {"ok": True, "ranked": [], "latency_ms": 0, "from_cache": True, "error": None}

    pool = candidates[:MAX_RESUMES]
    key = _hash_key(jd_text, pool)

    if use_cache and cache is not None and key in cache:
        resp = cache[key]
        ranked = [r for r in resp.get("ranked", []) if int(r.get("score", 0)) >= min_score][:top_k]
        return {"ok": True, "ranked": ranked, "latency_ms": 0, "from_cache": True, "error": None}

    payload = {
        "model": model or OPENAI_MODEL,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": _build_prompt(jd_text, pool),
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key or OPENAI_API_KEY}",
    }

    t0 = time.time()
    try:
        r = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        latency_ms = int((time.time() - t0) * 1000)
        if r.status_code != 200:
            return {"ok": False, "ranked": [], "latency_ms": latency_ms, "from_cache": False,
                    "error": f"HTTP {r.status_code}: {r.text[:200]}"}

        data = r.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)

        ranked = parsed.get("ranked", [])
        pool_ids = {str(c.get("resume_id") or c.get("id")) for c in pool}
        ranked = [r for r in ranked if str(r.get("resume_id")) in pool_ids]
        ranked = [r for r in ranked if int(r.get("score", 0)) >= min_score][:top_k]

        out = {"ok": True, "ranked": ranked, "latency_ms": latency_ms, "from_cache": False, "error": None}
        if use_cache and cache is not None:
            cache[key] = out
        return out

    except Exception as e:
        latency_ms = int((time.time() - t0) * 1000)
        return {"ok": False, "ranked": [], "latency_ms": latency_ms, "from_cache": False, "error": str(e)}
