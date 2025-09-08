import json
import logging
import re
import hashlib
from typing import Dict, List, Tuple

from openai import OpenAI
from django_core.config import Config

logger = logging.getLogger(__name__)

# Config
MAX_CHUNKS_FOR_LLM = 80
CHUNK_TEXT_CHAR_LIMIT = 500
MODEL_FALLBACK = "gpt-3.5-turbo"
PLACEHOLDER_PREFIX = "Candidate "

DISALLOWED_NAME_WORDS = {
    "applications","duration","projects","project","experience","education","summary","profile",
    "objective","android","developer","engineer","flipkart","responsibilities","analysis","skills",
    "career","work","history","personal","details","professional","roles","responsibility","role",
    "contact","address","mobile","phone","email","curriculum","vitae","resume"
}

def _shorten_text(t: str, limit: int) -> str:
    if not t:
        return ""
    t = t.replace("\n", " ").strip()
    return t[:limit] + ("..." if len(t) > limit else "")

def _hash_sig(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]

def _build_chunk_payload(raw_chunks: List[Dict]) -> List[Dict]:
    ordered = sorted(
        enumerate(raw_chunks),
        key=lambda kv: float(kv[1].get("score", 0.0)),
        reverse=True,
    )[:MAX_CHUNKS_FOR_LLM]
    compact = []
    for new_idx, (orig_idx, ch) in enumerate(ordered):
        text = ch.get("text") or ch.get("content") or ""
        snippet = _shorten_text(text, CHUNK_TEXT_CHAR_LIMIT)
        compact.append(
            {
                "chunk_id": new_idx,
                "orig_index": orig_idx,
                "file_name": ch.get("file_name") or "",
                "score": float(ch.get("score", 0.0)),
                "sig": _hash_sig(snippet[:120]),
                "text": snippet,
            }
        )
    return compact

PROMPT_TEMPLATE = """You are an AI assistant that groups and ranks resume chunks for a job description.

Goals:
1. Cluster chunks belonging to the same human candidate (each cluster = one person).
2. Infer the real human name (avoid company names / section headers / skill terms).
3. ONLY if a real personal name cannot be found, use a unique placeholder like "Candidate A", "Candidate B", etc.
4. Rank candidates by suitability for the job description.
5. Provide BOTH:
   - reason: short objective justification (<=35 words).
   - selection_summary: a concise summary of the candidate’s most relevant experience & skills (<=25 words, no personal info).
6. Output strictly valid JSON array (no comments, no markdown).

Job Description:
{jd}

Resume Chunks (sorted by relevance):
{chunks}

Output JSON array format (no trailing commas):

[
  {{
    "candidate_id": "<short unique id>",
    "name": "<Human Name or Candidate A if none>",
    "rank": 1,
    "reason": "<<=35 words justification>",
    "selection_summary": "<<=25 words summary of relevant experience>",
    "chunk_ids": [0,5,7],
    "reference_guess": "<optional filename stem or ''>"
  }},
  ...
]

Rules:
- rank starts at 1, increments by 1, no duplicates.
- Return at most {top_k} candidates (fewer if fewer clusters).
- Each chunk_id appears in exactly ONE candidate.
- Do not hallucinate data not in chunks.
- Avoid using these alone as name: {bad_words}.
- If multiple names appear in one cluster pick the most plausible full personal name (1–3 words).
- No extra fields beyond specified keys.
- No markdown fences, code blocks, or commentary.
"""

def _build_prompt(job_description: str, compact_chunks: List[Dict], top_k: int) -> str:
    chunk_lines = []
    for c in compact_chunks:
        chunk_lines.append(
            f"- chunk_id={c['chunk_id']} | orig_index={c['orig_index']} | score={c['score']:.2f} | file_name={c['file_name'] or 'NA'} | sig={c['sig']} | text=\"{c['text']}\""
        )
    return PROMPT_TEMPLATE.format(
        jd=job_description.strip(),
        chunks="\n".join(chunk_lines),
        top_k=top_k,
        bad_words=sorted(DISALLOWED_NAME_WORDS),
    )

def _clean_llm_output(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.strip()
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = raw.replace("```", "")
    raw = re.sub(r",\s*]", "]", raw)
    return raw

def _parse_output(raw: str) -> List[Dict]:
    cleaned = _clean_llm_output(raw)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    m = re.search(r"\[\s*{.*}\s*\]", cleaned, re.DOTALL)
    if m:
        seg = re.sub(r",\s*]", "]", m.group(0))
        try:
            data = json.loads(seg)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    raise ValueError("Failed to parse LLM grouping JSON.")

def _sanitize_name(name: str) -> Tuple[str, bool]:
    if not name:
        return "", True
    raw = name.strip()
    if not raw:
        return "", True
    lower = raw.lower()
    if lower in DISALLOWED_NAME_WORDS or len(raw) < 2:
        return "", True
    if re.fullmatch(r"[A-Z ]{3,}", raw):  # screams heading
        return "", True
    if any(tok.lower() in DISALLOWED_NAME_WORDS for tok in raw.split()):
        # allow multi-word if first two look like proper nouns
        tokens = raw.split()
        ok_tokens = [t for t in tokens if t[0].isupper()]
        if len(ok_tokens) == 0:
            return "", True
    return raw, False

def _finalize_placeholder(counter: int) -> str:
    return f"{PLACEHOLDER_PREFIX}{chr(ord('A') + counter)}"

def group_and_rank_with_llm(
    job_description: str,
    raw_chunks: List[Dict],
    top_k: int,
) -> Tuple[List[Dict], Dict]:
    api_key = Config.OPEN_AI_KEY
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")

    model = Config.GPT_3_MODEL or MODEL_FALLBACK
    compact = _build_chunk_payload(raw_chunks)
    if not compact:
        return [], {"chunks_sent": 0}

    prompt = _build_prompt(job_description, compact, top_k)
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.15,
        max_tokens=1300,
    )
    raw = response.choices[0].message.content or ""

    try:
        parsed = _parse_output(raw)
    except Exception as e:
        logger.error("LLM grouping parse failed: %s | Raw first 800 chars: %s", e, raw[:800])
        raise

    compact_id_to_raw_index = {c["chunk_id"]: c["orig_index"] for c in compact}
    used_compact_ids = set()
    placeholder_counter = 0
    out: List[Dict] = []

    for obj in parsed:
        if not isinstance(obj, dict):
            continue
        cids = obj.get("chunk_ids")
        if not isinstance(cids, list) or not cids:
            continue
        filtered = []
        for cid in cids:
            if isinstance(cid, int) and cid in compact_id_to_raw_index and cid not in used_compact_ids:
                filtered.append(cid)
                used_compact_ids.add(cid)
        if not filtered:
            continue

        raw_name = str(obj.get("name", "")).strip()
        clean_name, invalid = _sanitize_name(raw_name)
        if invalid:
            clean_name = _finalize_placeholder(placeholder_counter)
            placeholder_counter += 1

        reason = str(obj.get("reason", "")).strip()
        if len(reason.split()) > 40:
            reason = " ".join(reason.split()[:40])

        sel_summary = str(obj.get("selection_summary", "")).strip()
        if not sel_summary:
            # fallback from reason
            sel_summary = " ".join(reason.split()[:25])
        if len(sel_summary.split()) > 25:
            sel_summary = " ".join(sel_summary.split()[:25])

        rank = obj.get("rank")
        try:
            rank = int(rank)
        except Exception:
            rank = 999

        out.append(
            {
                "candidate_id": str(obj.get("candidate_id") or f"cand_{len(out)+1}"),
                "candidate_name": clean_name,
                "candidate_name_source": "llm" if not invalid else "placeholder",
                "chunk_ids_compact": filtered,
                "raw_chunk_indices": [compact_id_to_raw_index[cid] for cid in filtered],
                "llm_rank": rank,
                "llm_reason": reason,
                "selection_summary": sel_summary,
                "reference_guess": str(obj.get("reference_guess") or "").strip(),
            }
        )

    out.sort(key=lambda c: c.get("llm_rank", 999))
    for i, c in enumerate(out, 1):
        c["llm_rank"] = i

    diagnostics = {
        "chunks_sent": len(compact),
        "raw_response_excerpt": raw[:400],
        "candidate_count": len(out),
    }
    return out[:top_k], diagnostics
