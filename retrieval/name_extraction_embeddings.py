import re
import math
from typing import List, Dict, Tuple, Optional
import numpy as np

# You will need sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Install sentence-transformers: pip install sentence-transformers")

NAME_PROTOTYPES = [
    "John David Smith",
    "Rahul Kumar",
    "Ayaan Mohammed",
    "Priya Singh",
    "Emily Johnson",
    "Mohammed Ayaan",
    "Nivas Mohan",
    "Sourav Pal"
]

NEG_PROTOTYPES = [
    "SOFTWARE ENGINEER",
    "ANDROID DEVELOPER",
    "FULL STACK DEVELOPER",
    "SKILLS",
    "SUMMARY",
    "OBJECTIVE",
    "AWS KAFKA SYSTEM",
    "PROFESSIONAL EXPERIENCE",
    "TECHNICAL SKILLS"
]

ROLE_OR_TECH = {
    "engineer","developer","development","architect","programmer","software",
    "android","full","stack","lead","senior","junior","data","cloud","aws","kafka",
    "skills","summary","objective","profile","experience","professional","resume",
    "curriculum","vitae","cv","technologies","technology","expertise","manager"
}

NAME_TOKEN_PATTERN = re.compile(r"^[A-Z][a-z][a-zA-Z'\-]{0,30}$")
INITIAL_PATTERN = re.compile(r"^[A-Z]\.?$")
EMAIL_REGEX = re.compile(r"[A-Za-z0-9.\-_+]+@[A-Za-z0-9\.\-]+\.[A-Za-z]{2,}")
PHONE_REGEX = re.compile(r"\+?\d[\d \-\(\)]{6,}\d")
FILENAME_STOP = {"resume","cv","profile","final","updated","new","latest"}

_model = None
_name_proto_vecs = None
_neg_proto_vecs = None

def _get_model():
    global _model, _name_proto_vecs, _neg_proto_vecs
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _name_proto_vecs = _model.encode(NAME_PROTOTYPES, normalize_embeddings=True)
        _neg_proto_vecs = _model.encode(NEG_PROTOTYPES, normalize_embeddings=True)
    return _model, _name_proto_vecs, _neg_proto_vecs

def extract_name_with_embeddings(
    chunk_texts: List[str],
    file_name: Optional[str] = None,
    max_lines_scan: int = 120,
    window_radius: int = 4
) -> Dict:
    """
    Returns dict:
    {
      'name': str,
      'confidence': float,
      'source': str,
      'candidates': [...debug per candidate...],
      'fallback_used': bool
    }
    """
    combined = "\n".join(chunk_texts)
    lines_raw = [l.rstrip() for l in combined.splitlines()]
    lines = []
    for l in lines_raw:
        l2 = re.sub(r"\s+", " ", l).strip()
        if l2:
            lines.append(l2)
        if len(lines) >= max_lines_scan:
            break

    emails = EMAIL_REGEX.findall(combined)
    email_tokens = set()
    for e in emails:
        local = e.split("@",1)[0]
        for t in re.split(r"[._+\-]", local):
            tl = t.lower()
            if 2 <= len(tl) <= 25 and tl not in {"gmail","yahoo","outlook","hotmail","email","mail"}:
                email_tokens.add(tl)

    phone_indices = []
    email_indices = []
    for i,l in enumerate(lines):
        if EMAIL_REGEX.search(l):
            email_indices.append(i)
        if PHONE_REGEX.search(l):
            phone_indices.append(i)

    # filename hint
    file_hint_tokens = set()
    file_name_hint = None
    if file_name:
        base = re.sub(r"\.[A-Za-z0-9]+$","",file_name)
        base = re.sub(r"[_\-]+"," ",base)
        base = re.sub(r"\d{2,}"," ",base)
        toks = [t for t in base.split() if 2 <= len(t) <= 25]
        toks = [t for t in toks if t.lower() not in FILENAME_STOP]
        cand = [t for t in toks if re.match(r"^[A-Z][a-z]+$", t)]
        if 1 < len(cand) <= 4:
            file_name_hint = " ".join(cand[:3])
            file_hint_tokens = {c.lower() for c in cand}

    anchor_indices = sorted(set(email_indices + phone_indices))
    candidate_map = {}  # line_index -> candidate line

    # Always include first 30 lines (or fewer)
    for idx,l in enumerate(lines[:30]):
        if _looks_name_candidate(l):
            candidate_map[idx] = l

    # Include window around anchors
    for a in anchor_indices:
        start = max(0, a - window_radius)
        end = min(len(lines), a + window_radius + 1)
        for idx in range(start, end):
            line = lines[idx]
            if _looks_name_candidate(line):
                candidate_map[idx] = line

    if not candidate_map and file_name_hint:
        return {
            "name": file_name_hint,
            "confidence": 0.60,
            "source": "file_name_hint_only",
            "candidates": [],
            "fallback_used": True
        }

    model, name_proto_vecs, neg_proto_vecs = _get_model()

    candidate_items = []
    texts = [candidate_map[i] for i in sorted(candidate_map)]
    if texts:
        cand_vecs = model.encode(texts, normalize_embeddings=True)
    else:
        cand_vecs = []

    for vec, idx in zip(cand_vecs, sorted(candidate_map)):
        line = candidate_map[idx]
        parts = line.split()
        tokens_clean = [p.strip(",;") for p in parts]

        # pattern score
        good = sum(1 for p in tokens_clean if NAME_TOKEN_PATTERN.match(p) or INITIAL_PATTERN.match(p))
        pattern_score = good / len(tokens_clean)

        role_hits = sum(1 for p in tokens_clean if p.lower() in ROLE_OR_TECH)
        role_frac = role_hits / len(tokens_clean)

        email_align = sum(1 for p in tokens_clean if p.lower().rstrip(".") in email_tokens) / len(tokens_clean)

        filename_align = 0.0
        if file_hint_tokens:
            filename_align = sum(1 for p in tokens_clean if p.lower() in file_hint_tokens)/ len(tokens_clean)

        proximity = _proximity_score(idx, anchor_indices)

        # embedding prototype sims
        name_sim = float(np.mean(np.dot(name_proto_vecs, vec)))
        neg_sim = float(np.max(np.dot(neg_proto_vecs, vec)))
        proto_margin = name_sim - neg_sim

        # Compress via sigmoid-ish
        proto_component = 1 / (1 + math.exp(-4 * proto_margin))  # 0..1

        # Penalize lines with digits
        digit_penalty = 0.15 if any(ch.isdigit() for ch in line) else 0.0
        long_penalty = 0.0
        if len(tokens_clean) > 5:
            long_penalty = 0.10

        final_score = (
            0.45 * proto_component +
            0.20 * pattern_score +
            0.12 * email_align +
            0.08 * proximity +
            0.08 * (1 - role_frac) +
            0.07 * filename_align
            - digit_penalty
            - long_penalty
        )

        candidate_items.append({
            "line_index": idx,
            "line": line,
            "tokens": tokens_clean,
            "pattern_score": round(pattern_score,4),
            "role_frac": round(role_frac,4),
            "email_align": round(email_align,4),
            "filename_align": round(filename_align,4),
            "proximity": round(proximity,4),
            "name_sim": round(name_sim,4),
            "neg_sim": round(neg_sim,4),
            "proto_margin": round(proto_margin,4),
            "proto_component": round(proto_component,4),
            "digit_penalty": digit_penalty,
            "long_penalty": long_penalty,
            "final_score": round(final_score,4),
        })

    if not candidate_items:
        if file_name_hint:
            return {
                "name": file_name_hint,
                "confidence": 0.55,
                "source": "file_name_hint_only",
                "candidates": [],
                "fallback_used": True
            }
        return {
            "name": "Candidate",
            "confidence": 0.30,
            "source": "fallback_none",
            "candidates": [],
            "fallback_used": True
        }

    # Pick best
    candidate_items.sort(key=lambda x: x["final_score"], reverse=True)
    best = candidate_items[0]
    name_candidate = _normalize_name_line(best["line"])
    generic = _is_generic(name_candidate)

    if generic and file_name_hint:
        name_candidate = file_name_hint
        best["final_score"] = max(best["final_score"], 0.60)

    confidence = 0.40 + min(1.0, best["final_score"]) * 0.53
    confidence = round(min(confidence, 0.94), 3)

    return {
        "name": name_candidate,
        "confidence": confidence,
        "source": "embedding_proto",
        "candidates": candidate_items,
        "file_name_hint": file_name_hint,
        "fallback_used": False
    }

def _looks_name_candidate(line: str) -> bool:
    # Quick structural check
    if len(line) > 70:
        return False
    if "|" in line and line.count("|") >= 2:
        return False
    if line.count(",") >= 2:
        return False
    tokens = [t.strip(",;") for t in line.split()]
    if not (2 <= len(tokens) <= 5):
        return False
    # At least two tokens with initial-cap pattern or initials
    good = sum(1 for t in tokens if NAME_TOKEN_PATTERN.match(t) or INITIAL_PATTERN.match(t))
    if good < 2:
        return False
    # Avoid heavy role heading (all tokens role terms)
    if all(t.lower() in ROLE_OR_TECH for t in tokens):
        return False
    return True

def _proximity_score(idx: int, anchors: List[int]) -> float:
    if not anchors:
        return 0.0
    dist = min(abs(idx - a) for a in anchors)
    return 1.0 / (1 + dist)  # 1 if same line, decays quickly

def _normalize_name_line(line: str) -> str:
    toks = [t.strip(",;") for t in line.split()]
    cleaned = []
    for t in toks:
        if NAME_TOKEN_PATTERN.match(t):
            cleaned.append(t)
        elif INITIAL_PATTERN.match(t):
            cleaned.append(t.rstrip("."))
        else:
            # Attempt salvage for ALL CAPS single word
            if t.isupper() and len(t) > 1:
                cleaned.append(t.capitalize())
    # Enforce length
    if 1 < len(cleaned) <= 5:
        return " ".join(cleaned)
    # fallback original
    return " ".join(toks)

def _is_generic(name: str) -> bool:
    low = name.lower()
    return low in {"candidate","profile","resume"} or any(
        low.startswith(x) for x in ["android","aws","kafka","system","email","skills"]
    )
