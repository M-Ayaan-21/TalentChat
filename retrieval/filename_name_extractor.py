import re
from typing import List, Dict, Set, Optional, Tuple

GENERIC_TOKENS = {
    "resume","cv","profile","final","updated","new","latest","draft","copy","version","sample",
    "doc","docx","pdf","rev","v1","v2","v3","v4","v5","old","first","second"
}

ROLE_OR_TECH = {
    "developer","engineer","engineering","programmer","architect","consultant","intern","student",
    "manager","specialist","administrator","admin","analyst","devops","data","android","ios",
    "full","stack","fullstack","backend","frontend","cloud","ai","ml","python","java","kotlin",
    "csharp","c++","cpp","node","react","flutter","django","spring","aws","gcp","azure",
    "security","qa","testing","automation","mobile"
}

NAME_TOKEN_PATTERN = re.compile(r"^[A-Z][a-zA-Z'\-]{1,30}$")
CAMEL_CASE_PATTERN = re.compile(r"[A-Z][a-z]+")
YEAR_PATTERN = re.compile(r"^(19|20)\d{2}$")
VERSION_PATTERN = re.compile(r"^v?\d{1,2}$", re.I)
DIGIT_STRIP = re.compile(r"\d+")

EMAIL_SPLIT = re.compile(r"[._+\-]")

def _split_filename_base(file_name: str) -> List[str]:
    base = file_name.rsplit("/",1)[-1]
    base = re.sub(r"\.[A-Za-z0-9]+$", "", base)
    base = re.sub(r"\([^)]*\)", " ", base)
    base = re.sub(r"[ _\-.]+", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base.split() if base else []

def _explode_camel_or_fused(token: str) -> List[str]:
    camel = CAMEL_CASE_PATTERN.findall(token)
    if camel and "".join(camel) == token:
        return camel
    if token.isupper() and len(token) > 8:
        mid = len(token)//2
        return [token[:mid].capitalize(), token[mid:].capitalize()]
    return [token]

def _normalize_tokens(raw_tokens: List[str]) -> List[str]:
    out = []
    for t in raw_tokens:
        t = DIGIT_STRIP.sub("", t)
        if not t:
            continue
        for sub in _explode_camel_or_fused(t):
            if sub.isupper() and len(sub) > 1:
                sub = sub.capitalize()
            if sub:
                out.append(sub)
    return out

def _filter_tokens(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        low = t.lower()
        if low in GENERIC_TOKENS: continue
        if low in ROLE_OR_TECH: continue
        if YEAR_PATTERN.match(t): continue
        if VERSION_PATTERN.match(t): continue
        out.append(t)
    return out

def _extract_email_tokens(emails: List[str]) -> Set[str]:
    toks = set()
    for e in emails:
        local = e.split("@",1)[0]
        for part in EMAIL_SPLIT.split(local):
            p = part.strip().lower()
            if 2 <= len(p) <= 25 and p not in {"gmail","email","mail","yahoo","outlook","hotmail"}:
                toks.add(p)
    return toks

def _score_tokens(tokens: List[str], email_tokens: Set[str]) -> Dict:
    if not tokens:
        return {"raw_score":0.0,"pattern_score":0,"len_score":0,"email_align":0,"lower_penalty":0}
    pattern = sum(1 for t in tokens if NAME_TOKEN_PATTERN.match(t) or (len(t)==1 and t.isupper())) / len(tokens)
    if len(tokens) == 1: length = 0.55
    elif len(tokens) in (2,3): length = 1.0
    elif len(tokens) == 4: length = 0.75
    else: length = 0.45
    email_align = 0.0
    if email_tokens:
        email_align = sum(1 for t in tokens if t.lower() in email_tokens)/len(tokens)
    lower_penalty = sum(1 for t in tokens if t.islower())/len(tokens)*0.2
    raw = 0.55*pattern + 0.15*length + 0.20*email_align - lower_penalty
    raw = max(0.0, min(1.0, raw))
    return {
        "raw_score": round(raw,4),
        "pattern_score": round(pattern,4),
        "len_score": round(length,4),
        "email_align": round(email_align,4),
        "lower_penalty": round(lower_penalty,4)
    }

def derive_name_from_filename(file_name: str, emails: List[str]) -> Dict:
    email_tokens = _extract_email_tokens(emails)
    raw_tokens = _split_filename_base(file_name)
    norm = _normalize_tokens(raw_tokens)
    filt = _filter_tokens(norm)
    candidate = filt or norm
    # If looks like LAST FIRST reorder:
    if len(candidate) >= 2 and candidate[0].isupper() and not candidate[1].isupper():
        candidate = candidate[1:] + candidate[:1]
    if len(candidate) > 4:
        candidate = candidate[:4]
    score_dbg = _score_tokens(candidate, email_tokens)
    if not candidate:
        return {"name": None,"confidence":0.0,"source":"filename","debug":score_dbg}
    name = " ".join(candidate)
    confidence = round(0.45 + score_dbg["raw_score"] * 0.45,3)
    return {"name":name,"confidence":confidence,"source":"filename","debug":score_dbg,"email_tokens":list(email_tokens)}
