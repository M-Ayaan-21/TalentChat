"""
question_generator.py

Unified question generation module with backward compatibility.

Public functions:
 - generate_candidate_questions(job_description: str, candidate_bundle: dict) -> List[str]
      New, richer interface (JD + candidate aggregated resume context).
 - generate_questions(...)
      Backward compatibility wrapper. Accepts legacy call shapes:
        a) generate_questions(candidate_name, summary)
        b) generate_questions(candidate_name, level, embeddings)
        c) generate_questions(job_description, candidate_bundle_dict)

Behavior:
 - If OPENAI key (Config.OPEN_AI_KEY) & openai package available -> uses OpenAI chat completion.
 - Else -> deterministic fallback list tailored to detected skills & summary.

candidate_bundle expected keys (for new flow):
   candidate_name
   selection_summary
   deterministic_summary
   aggregated_text
   matched_chunks (list[{content, score}, ...])
"""

from __future__ import annotations
from typing import List, Dict, Any, Sequence, Optional, Union
import re
import json
import logging

from django_core.config import Config

logger = logging.getLogger(__name__)

# ------------------------------------------------------
# OpenAI availability
# ------------------------------------------------------
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# ------------------------------------------------------
# Skill / keyword extraction (simple heuristic)
# ------------------------------------------------------
STOPWORDS = {
    "the","and","with","from","that","this","have","has","for","you","are","was","your",
    "our","their","they","will","can","but","not","use","used","using","into","all","any",
    "each","other","than","over","off","very","more","most","much","many","such","about",
    "able","also","been","being","were","them","then","there","these","those","some","just",
    "well","work","role","team","project","projects","experience","years","year","responsible",
    "responsibilities","solution","solutions","based"
}

TECH_WHITELIST = {
    "python","java","javascript","typescript","node","nodejs","react","reactjs","reactjs",
    "django","flask","fastapi","spring","hibernate","go","golang","ruby","rails","rust",
    "c","c++","c#","dotnet",".net","php","laravel","kotlin","swift","android","ios",
    "ml","ai","nlp","llm","pytorch","tensorflow","keras","sklearn","scikit","aws","gcp",
    "azure","docker","kubernetes","kubeflow","airflow","hadoop","spark","hive","kafka",
    "mysql","postgres","postgresql","redis","mongodb","elasticsearch","graphql","rest",
    "microservices","terraform","ansible","jenkins","git","gitlab","ci","cd","serverless",
    "lambda","s3","athena","glue","snowflake","bigquery","supabase","firebase","nextjs",
    "vue","angular","svelte","tailwind","storybook","webpack","vite","babel"
}


def _extract_skill_tokens(text: str, limit: int = 14) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+\-#]{1,}", text)
    freq: Dict[str, int] = {}
    for raw in tokens:
        t = raw.lower()
        if len(t) < 2:
            continue
        if t in STOPWORDS:
            continue
        freq[t] = freq.get(t, 0) + 1

    # Boost if in TECH_WHITELIST
    scored = []
    for w, f in freq.items():
        base = f
        if w in TECH_WHITELIST:
            base += 2
        scored.append((w, base))

    scored.sort(key=lambda x: (-x[1], x[0]))
    out = [w for w, _ in scored[:limit]]
    return out


# ------------------------------------------------------
# Fallback deterministic questions
# ------------------------------------------------------
def _fallback_questions(candidate_name: str, jd: str, bundle: Dict[str, Any]) -> List[str]:
    agg = bundle.get("aggregated_text", "") or ""
    summary = bundle.get("selection_summary") or bundle.get("deterministic_summary") or ""
    text_for_skills = " ".join([
        agg[:3000],
        jd[:2000],
        summary[:800]
    ])
    skills = _extract_skill_tokens(text_for_skills)
    top_skills = ", ".join(skills[:6]) if skills else "the relevant technologies"

    q: List[str] = [
        f"Describe a recent project where you applied {top_skills}. What was the hardest technical challenge?",
        "Walk me through a complex architectural or design decision you influenced. Why was it chosen?",
        "Pick a performance bottleneck you solved: what metrics improved and how?",
        "How do you ensure code quality (tests, reviews, automation) under tight deadlines?",
        "Explain a time you refactored legacy or unstructured code into something maintainable.",
        "Give an example where collaboration with non‑engineering stakeholders changed your implementation approach.",
        "Describe a failure or production incident you handled; what changed afterward?",
        f"In your resume summary you mention: '{summary[:90]}...'. Provide more measurable impact or metrics.",
        "Select one bullet from your experience and break down the exact stack, data flow, and trade‑offs.",
        f"If hired for this JD, what would your 60‑day technical roadmap focus on first?"
    ]
    return q[:10]


# ------------------------------------------------------
# OpenAI path
# ------------------------------------------------------
def _openai_questions(jd: str, bundle: Dict[str, Any]) -> List[str]:
    if not (OPENAI_AVAILABLE and Config.OPEN_AI_KEY):
        return []

    client = OpenAI(api_key=Config.OPEN_AI_KEY)

    candidate_name = bundle.get("candidate_name") or "Candidate"
    jd_trim = jd.strip()[:2500]
    agg_text = (bundle.get("aggregated_text") or "")[:3500]
    sel_sum = (
        bundle.get("selection_summary")
        or bundle.get("deterministic_summary")
        or ""
    )
    chunks = bundle.get("matched_chunks") or []
    chunk_preview = "\n---\n".join((c.get("content") or "")[:320] for c in chunks[:4])

    prompt = f"""
You are an AI generating INTERVIEW QUESTIONS tailored to a single candidate and the provided job description.

Job Description (truncated):
\"\"\"{jd_trim}\"\"\"

Candidate Name: {candidate_name}
Candidate Summary: {sel_sum}
Resume Aggregated Text:
\"\"\"{agg_text}\"\"\"

Key Extracted Chunks:
{chunk_preview}

Task:
Generate exactly 10 unique questions:
- 5 deep technical / architectural / skill questions grounded in resume details.
- 3 behavioral / collaboration / problem-solving questions referencing context when possible.
- 2 resume-specific drill-down questions focusing on measurable impact / metrics / trade-offs.

Rules:
- NO numbering like "1." Just a pure JSON array of strings.
- Each question must be unique, specific, and not generic filler.
- Avoid repeating candidate name in every question.
- Questions must be actionable (ask for specifics, metrics, reasoning, trade-offs).

Return ONLY a valid JSON array of strings.
"""

    try:
        resp = client.chat.completions.create(
            model=Config.GPT_4_MODEL or Config.GPT_3_MODEL or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.55,
            max_tokens=900,
        )
        raw = resp.choices[0].message.content.strip()
        match = re.search(r"\[\s*\".*?\"\s*\]", raw, re.DOTALL)
        if match:
            arr = json.loads(match.group(0))
            if isinstance(arr, list):
                cleaned = []
                for q in arr:
                    if isinstance(q, str):
                        qt = q.strip()
                        if qt and len(qt) > 8:
                            cleaned.append(qt)
                if cleaned:
                    return cleaned[:10]
    except Exception as e:
        logger.warning("OpenAI question generation failed: %s", e)

    return []  # fallback will happen upstream


# ------------------------------------------------------
# Public: new canonical generator
# ------------------------------------------------------
def generate_candidate_questions(job_description: str, candidate_bundle: Dict[str, Any]) -> List[str]:
    candidate_name = candidate_bundle.get("candidate_name") or "Candidate"

    # Try OpenAI path
    ai_qs = _openai_questions(job_description, candidate_bundle)
    if ai_qs:
        return ai_qs

    # Fallback deterministic
    return _fallback_questions(candidate_name, job_description, candidate_bundle)


# ------------------------------------------------------
# Backward Compatibility Wrapper
# ------------------------------------------------------
def generate_questions(*args, **kwargs) -> List[str]:
    """
    Backward compatible wrapper so legacy imports (generate_questions) still work.

    Supported call patterns:
      1) generate_questions(candidate_name, summary)
         -> returns generic questions using candidate_name + summary

      2) generate_questions(candidate_name, level, embeddings)
         -> ignores level/embeddings beyond name; returns generic fallback

      3) generate_questions(job_description, candidate_bundle_dict)
         -> if candidate_bundle_dict appears to be a dict with aggregated_text,
            defers to generate_candidate_questions(job_description, candidate_bundle)

    Any unrecognized pattern -> generic fallback with minimal context.
    """
    if not args:
        return _fallback_questions("Candidate", "", {})

    # Pattern 3 detection:
    if len(args) >= 2 and isinstance(args[1], dict) and (
        "aggregated_text" in args[1] or "matched_chunks" in args[1]
    ):
        # treat as new style
        jd = args[0]
        bundle = args[1]
        return generate_candidate_questions(jd, bundle)

    # Pattern 1: (candidate_name, summary)
    if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], str):
        candidate_name, summary = args
        bundle = {
            "candidate_name": candidate_name,
            "selection_summary": summary,
            "deterministic_summary": summary,
            "aggregated_text": summary,
            "matched_chunks": [],
        }
        return generate_candidate_questions(summary, bundle)  # reuse summary as pseudo JD

    # Pattern 2: (candidate_name, level, embeddings)
    if len(args) >= 3 and isinstance(args[0], str):
        candidate_name = args[0]
        level = args[1]
        # embeddings = args[2]  # unused fallback
        pseudo_jd = f"Candidate level: {level}"
        bundle = {
            "candidate_name": candidate_name,
            "selection_summary": pseudo_jd,
            "deterministic_summary": pseudo_jd,
            "aggregated_text": pseudo_jd,
            "matched_chunks": [],
        }
        return generate_candidate_questions(pseudo_jd, bundle)

    # If kwargs contain job_description & candidate_bundle
    if "job_description" in kwargs and "candidate_bundle" in kwargs:
        return generate_candidate_questions(kwargs["job_description"], kwargs["candidate_bundle"])

    # Final fallback
    return _fallback_questions("Candidate", "", {})


__all__ = [
    "generate_candidate_questions",
    "generate_questions",
]
