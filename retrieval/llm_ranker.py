import json
import re
from typing import List, Dict, Tuple
from openai import OpenAI
from django_core.config import Config

RANK_PROMPT_TEMPLATE = """You are a technical recruiter assistant.
Rank the candidates for the given job description using their summarized profiles.

Job Description:
{jd}

Candidate Summaries:
{summaries}

Return ONLY strict JSON (array) like:
[
  {{"candidate_id":"<id>","rank":1,"reason":"<=35 words","selection_summary":"<=25 words"}}
]

Rules:
- Up to {top_k} candidates (or fewer).
- rank starts at 1, increments by 1.
- No extra fields. No commentary. No trailing commas.
"""

def build_rank_prompt(job_description: str, summaries: List[Dict], top_k: int) -> str:
    lines = []
    for s in summaries:
        lines.append(
            f"- candidate_id={s['candidate_id']} | name={s['candidate_name']} | score={s['score']:.2f} | summary=\"{s['summary']}\""
        )
    return RANK_PROMPT_TEMPLATE.format(
        jd=job_description.strip(),
        summaries="\n".join(lines),
        top_k=top_k
    )

def _clean(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = raw.replace("```", "")
    raw = re.sub(r",\s*]", "]", raw)
    return raw

def _parse(raw: str) -> List[Dict]:
    cleaned = _clean(raw)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    m = re.search(r"\[\s*{.*}\s*\]", cleaned, re.DOTALL)
    if m:
        segment = re.sub(r",\s*]", "]", m.group(0))
        try:
            data = json.loads(segment)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    raise ValueError("Failed to parse ranking JSON.")

def llm_rank(job_description: str, summaries: List[Dict], top_k: int) -> Tuple[List[Dict], str]:
    api_key = Config.OPEN_AI_KEY
    if not api_key:
        return [], "missing_api_key"

    client = OpenAI(api_key=api_key)
    model = Config.GPT_3_MODEL or "gpt-3.5-turbo"
    prompt = build_rank_prompt(job_description, summaries, top_k)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=900,
    )
    raw = resp.choices[0].message.content or ""
    try:
        parsed = _parse(raw)
        return parsed, ""
    except Exception as e:
        return [], f"parse_error: {e}"
