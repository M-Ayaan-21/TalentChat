"""
Content Retrieval (v18.1)
 - Builds on prior v18
 - Improved human name extraction & reconstruction
 - Fallback resume URL logic & aggregated_text for downstream question generation
 - Slightly adjusted ranking weights
"""

from __future__ import annotations
import datetime
import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, List, Optional, Set, Tuple

from common.utils import send_request
from django_core.config import Config

logger = logging.getLogger(__name__)

CONFIG = {
    "USE_DETERMINISTIC_GROUPING": True,
    "ENABLE_TTL_CACHE": True,
    "CACHE_TTL_SECONDS": 180,
    "CACHE_MAX_ENTRIES": 128,

    # Ranking weights (slight shift vs v18)
    "WEIGHT_RAW_SCORE": 0.40,
    "WEIGHT_KEYWORD_COVERAGE": 0.35,
    "WEIGHT_BONUSES": 0.25,

    "BONUS_HAS_EMAIL": 0.06,
    "BONUS_HAS_PHONE": 0.04,
    "BONUS_MULTI_CHUNK_STEP": 0.03,
    "BONUS_MULTI_CHUNK_MAX": 0.12,
    "BONUS_LONG_SUMMARY": 0.05,

    "PENALTY_HEADING_NAME": -0.15,
    "PENALTY_LOCATION_NAME": -0.08,
    "PENALTY_COMPANY_NAME": -0.10,
    "PENALTY_SINGLE_CHUNK_HEADING": -0.08,

    "PRUNE_SCORE_BELOW_FRAC_MEDIAN": 0.40,
    "PRUNE_REQUIRE_MIN_CHUNKS_IF_NO_CONTACT": 2,

    "MAX_TOP_K": 15,
    "MAX_GROUPS_PROCESS": 160,

    "MAX_NAME_LEN": 42,
    "MIN_KEYWORD_LENGTH": 3,
    "MAX_KEYWORDS": 60,

    "INCLUDE_NAME_DEBUG": True,
    "INCLUDE_GROUP_DEBUG": False,
}

EMAIL_RE = re.compile(r"[A-Za-z0-9.\-_+]+@[A-Za-z0-9\.\-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+\d{1,3}[-\s]?)?\d[\d\s().\-]{5,}\d")
NAME_LINE_RE = re.compile(r"^\s*([A-Z][A-Za-z'’\-]{1,30})(?:\s+([A-Z][A-Za-z'’\-]{1,30}|[A-Z]\.)){0,3}\s*$")

STOP_HEADING_TOKENS = {
    "skills","technical skills","achievements","achievement","abilities","ability",
    "summary","professional summary","profile","objective","career objective",
    "education","experience","work experience","employment history","work history",
    "projects","project","certifications","certification","tools","version control tools",
    "technologies","technology","responsibilities","roles & responsibilities",
    "roles and responsibilities","awards","strengths","interests","hobbies","contact",
    "curriculum vitae","resume","cv","about","about me"
}

LOCATION_WORDS = {
    "india","bengaluru","bangalore","mumbai","pune","delhi","hyderabad",
    "chennai","usa","canada","germany","france","london","singapore",
    "australia","remote","california","texas","new","york","seattle"
}

COMPANY_HINTS = {
    "pvt","private","ltd","limited","llc","inc","corp","corporation","studios","solutions",
    "technologies","technology","systems","labs","software","services","consulting","group"
}

STOPWORDS = {
    "the","and","with","from","that","this","have","has","for","you","are","was",
    "your","our","their","they","will","can","but","not","use","used","using","into",
    "all","any","each","other","than","over","off","very","more","most","much","many",
    "such","about","able","also","been","being","were","them","then","there","these",
    "those","some","just","well"
}

class _TTLCache:
    def __init__(self, max_entries: int, ttl_seconds: int):
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def _purge(self):
        now = time.time()
        for k in list(self._store.keys()):
            ts, _ = self._store[k]
            if now - ts > self.ttl:
                self._store.pop(k, None)
        if len(self._store) > self.max_entries:
            for k in list(self._store.keys())[: len(self._store) - self.max_entries]:
                self._store.pop(k, None)

    def get(self, key: str):
        with self._lock:
            self._purge()
            v = self._store.get(key)
            if not v:
                return None
            ts, val = v
            if time.time() - ts > self.ttl:
                self._store.pop(key, None)
                return None
            return val

    def set(self, key: str, value: Any):
        with self._lock:
            self._purge()
            self._store[key] = (time.time(), value)

_global_cache = _TTLCache(CONFIG["CACHE_MAX_ENTRIES"], CONFIG["CACHE_TTL_SECONDS"]) if CONFIG["ENABLE_TTL_CACHE"] else None

# --------------------- Helpers ---------------------
def _hash_jd(jd: str) -> str:
    return hashlib.sha1(jd.encode("utf-8")).hexdigest()[:16]

def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    return text[:limit] + ("..." if len(text) > limit else "")

def _tokenize_lower(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9+\-]{2,}", text.lower())

def _extract_keywords(text: str) -> List[str]:
    tokens = _tokenize_lower(text)
    freq: Dict[str,int] = {}
    for t in tokens:
        if len(t) < CONFIG["MIN_KEYWORD_LENGTH"]: continue
        if t in STOPWORDS: continue
        freq[t] = freq.get(t, 0) + 1
    ordered = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w,_ in ordered[:CONFIG["MAX_KEYWORDS"]]]

def _is_heading_like(s: str) -> bool:
    return s.strip().lower() in STOP_HEADING_TOKENS

def _is_location_line(s: str) -> bool:
    t = s.strip().lower()
    if "," in t:
        parts = [p.strip() for p in t.split(",") if p.strip()]
        if 1 <= len(parts) <= 3 and any(p in LOCATION_WORDS for p in parts):
            return True
    toks = t.split()
    if 1 <= len(toks) <= 4 and any(tok in LOCATION_WORDS for tok in toks):
        return True
    return False

def _looks_company_name(s: str) -> bool:
    l = s.lower()
    words = l.split()
    if any(w in COMPANY_HINTS for w in words):
        return True
    if len(words) <= 5 and any(w.isupper() and len(w) > 2 for w in words):
        return True
    return False

def _email_local_part_to_name(email: str) -> Optional[str]:
    local = email.split("@",1)[0]
    local = re.sub(r"\d+$","", local)  # strip trailing numbers
    parts = re.split(r"[._\-+]+", local)
    parts = [p for p in parts if p and p not in {"mail","resume","cv"}]
    if len(parts) < 2: return None
    tokens = [p[:1].upper()+p[1:].lower() for p in parts[:3]]
    candidate = " ".join(tokens)
    if 3 <= len(candidate) <= CONFIG["MAX_NAME_LEN"]:
        return candidate
    return None

def _split_all_caps(line: str) -> str:
    # ABOUT ME -> About Me
    words = line.split()
    if not words: return line
    transformed = []
    for w in words:
        if w.isupper() and len(w) > 1:
            transformed.append(w[0] + w[1:].lower())
        else:
            transformed.append(w)
    return " ".join(transformed)

def _extract_contacts(chunks: List[Dict]) -> Tuple[Set[str], Set[str]]:
    emails, phones = set(), set()
    for ch in chunks:
        txt = (ch.get("text") or ch.get("content") or "")
        for e in EMAIL_RE.findall(txt):
            emails.add(e.lower())
        for p in PHONE_RE.findall(txt):
            norm = re.sub(r"\D","", p)
            if 7 <= len(norm) <= 14:
                phones.add(norm)
    return emails, phones

def _scan_possible_name_lines(chunks: List[Dict], max_lines=30) -> List[str]:
    lines = []
    for ch in chunks[:6]:
        raw = (ch.get("text") or ch.get("content") or "")
        for ln in raw.splitlines():
            ln = ln.strip()
            if ln:
                lines.append(ln)
                if len(lines) >= max_lines:
                    break
        if len(lines) >= max_lines:
            break
    return lines

def _find_human_name_initial(chunks: List[Dict]) -> Optional[str]:
    ordered = sorted(chunks, key=lambda c: float(c.get("score",0.0)), reverse=True)
    for ch in ordered:
        txt = (ch.get("text") or ch.get("content") or "")
        line = txt.strip().split("\n",1)[0][:70]
        if NAME_LINE_RE.match(line) and " " in line:
            return line.strip()
    return None

def _multi_pass_name(initial: str, chunks: List[Dict], emails: Set[str], phones: Set[str], debug: Dict[str,Any]) -> Tuple[str,float,str]:
    name = (initial or "").strip()
    if name:
        # Avoid simple headings or location lines
        low = name.lower()
        if not (_is_heading_like(low) or _is_location_line(low) or _looks_company_name(low)):
            if 1 < len(name.split()) <= 4:
                return name, 0.55, "initial"
    # Scan lines around email/phone
    lines = _scan_possible_name_lines(chunks)
    for i, line in enumerate(lines):
        low = line.lower()
        if len(line) > 70: continue
        if _is_heading_like(low) or _is_location_line(low): continue
        if _looks_company_name(low): continue
        if NAME_LINE_RE.match(line) and " " in line and 1 < len(line.split()) <= 4:
            debug["scan_name"] = line
            return line.strip(), 0.70, "scan_regex"
        # For an email line -> previous line might be name
        if EMAIL_RE.search(line) and i>0:
            prev = lines[i-1].strip()
            if prev and not (_is_heading_like(prev.lower()) or _looks_company_name(prev.lower())) and 1 <= len(prev.split()) <= 5:
                nice = _split_all_caps(prev)
                debug["email_prev_line_name"] = nice
                return nice, 0.65, "prev_line_email"
    # Email local part
    for e in emails:
        nm = _email_local_part_to_name(e)
        if nm:
            debug["email_local_part_name"] = nm
            return nm, 0.62, "email_local_part"
    # Last resort: fix all caps initial
    if name and name.isupper():
        fixed = _split_all_caps(name)
        if fixed.lower() not in STOP_HEADING_TOKENS:
            debug["caps_fix_name"] = fixed
            return fixed, 0.45, "caps_fix"
    return "Candidate", 0.30, "fallback"

def _keyword_overlap(keywords: List[str], text: str) -> float:
    if not keywords:
        return 0.0
    tokens = set(_tokenize_lower(text))
    hits = sum(1 for kw in keywords if kw in tokens)
    return min(1.0, hits / max(4, len(keywords)))

def _canonical_signature(chunks: List[Dict]) -> str:
    ordered = sorted(chunks, key=lambda c: float(c.get("score",0.0)), reverse=True)
    if not ordered: return ""
    base = (ordered[0].get("text") or ordered[0].get("content") or "")[:120].lower()
    base = re.sub(r"[^a-z0-9]+","", base)
    return hashlib.md5(base.encode()).hexdigest()[:16]

def _join_url(base: str, rel: str) -> str:
    if not rel:
        return ""
    if rel.startswith(("http://","https://")):
        return rel
    if not rel.startswith("/"):
        rel = "/"+rel
    return base.rstrip("/") + rel

def _url_variants(domain_url: str, reference: Optional[str], file_name: Optional[str]) -> List[str]:
    out: List[str] = []
    root_domain = None
    if "/be" in domain_url.rstrip("/"):
        root_domain = domain_url.rstrip("/").rsplit("/be",1)[0]
    def add(u: str):
        if u and u not in out:
            out.append(u)
    if reference:
        add(_join_url(domain_url, reference))
        if root_domain: add(_join_url(root_domain, reference))
    if file_name and re.search(r"\.(pdf|docx?|rtf)$", file_name, re.I):
        guess = f"/media/users/resources/{file_name}"
        add(_join_url(domain_url, guess))
        if root_domain: add(_join_url(root_domain, guess))
    return out

# --------------------- Data Class ---------------------
@dataclass
class Candidate:
    candidate_id: str
    group_ids: List[str]
    name: str
    name_conf: float
    name_source: str
    name_debug: Dict[str,Any]
    chunks: List[Dict]
    file_names: List[str]
    reference_file: Optional[str]
    raw_score_sum: float
    summary: str
    selection_summary: str
    aggregated_text: str
    emails: Set[str]
    phones: Set[str]
    resume_url: str
    resume_url_alts: List[str]

    raw_norm: float = 0.0
    coverage_score: float = 0.0
    bonus_total: float = 0.0
    composite: float = 0.0

    heading_like: bool = False
    location_like: bool = False
    company_like: bool = False
    pruned: bool = False
    prune_reason: Optional[str] = None
    final_rank: Optional[int] = None

    def to_json(self, include_debug=True):
        data = {
            "candidate_id": self.candidate_id,
            "merged_from": self.group_ids,
            "candidate_name": self.name,
            "candidate_name_confidence": self.name_conf,
            "candidate_name_source": self.name_source,
            "raw_score_sum": self.raw_score_sum,
            "raw_norm": round(self.raw_norm,4),
            "coverage_score": round(self.coverage_score,4),
            "bonus_total": round(self.bonus_total,4),
            "composite_rank_score": round(self.composite,4),
            "selection_summary": self.selection_summary,
            "deterministic_summary": self.summary,
            "aggregated_text": self.aggregated_text,
            "matched_chunks": self.chunks,
            "resume_url": self.resume_url,
            "resume_url_alternatives": self.resume_url_alts,
            "pruned": self.pruned,
            "pruned_reason": self.prune_reason,
            "heading_like": self.heading_like,
            "location_like": self.location_like,
            "company_like": self.company_like,
            "final_rank": self.final_rank,
            "contact_emails": sorted(self.emails),
            "contact_phones": sorted(self.phones),
        }
        if include_debug:
            data["candidate_name_debug"] = self.name_debug
        return data

# --------------------- Grouping ---------------------
def _det_group(chunks: List[Dict], references: List[str]) -> Dict[str, Dict]:
    if CONFIG["USE_DETERMINISTIC_GROUPING"]:
        try:
            from retrieval.deterministic_grouping import group_chunks  # type: ignore
            return group_chunks(chunks, references)
        except Exception:
            logger.exception("deterministic grouping failed; fallback per-chunk.")
    groups = {}
    for i, ch in enumerate(chunks):
        ref = references[i] if i < len(references) else None
        groups[f"chunk::{i}"] = {"chunks":[ch], "reference": ref}
    return groups

def _initial_name(chunks: List[Dict]) -> Tuple[str,float,str,Dict]:
    try:
        from retrieval.deterministic_grouping import extract_candidate_name  # type: ignore
        return extract_candidate_name(chunks)
    except Exception:
        pass
    guess = _find_human_name_initial(chunks)
    if guess:
        return guess, 0.55, "regex_line", {"note":"regex_line"}
    return "Candidate", 0.30, "fallback", {"note":"fallback"}

# --------------------- Build Candidates ---------------------
def _build_candidates(groups: Dict[str,Dict], references: List[str], domain_url: str) -> List[Candidate]:
    out: List[Candidate] = []
    for gid, gdata in groups.items():
        gchunks = gdata["chunks"]
        if not gchunks:
            continue
        init_name, init_conf, init_source, init_debug = _initial_name(gchunks)
        emails, phones = _extract_contacts(gchunks)
        name, new_conf, new_source = _multi_pass_name(init_name, gchunks, emails, phones, init_debug)
        if new_conf >= 0:
            init_conf, init_source = new_conf, new_source

        ordered = sorted(gchunks, key=lambda c: float(c.get("score",0.0)), reverse=True)
        total_score = sum(float(c.get("score",0.0)) for c in ordered)
        matched = []
        file_names = set()
        text_for_agg_parts = []
        agg_len = 0
        for ch in ordered[:15]:
            sc = float(ch.get("score",0.0))
            txt = (ch.get("text") or ch.get("content") or "")
            matched.append({"content": _truncate(txt, 1000), "score": sc})
            if txt and agg_len < 3500:
                snippet = txt.strip()
                text_for_agg_parts.append(snippet)
                agg_len += len(snippet)
            if ch.get("file_name"):
                file_names.add(ch["file_name"])

        aggregated_text = "\n\n".join(text_for_agg_parts)[:4000]
        summary = _truncate(" | ".join(p[:170] for p in text_for_agg_parts[:8]), 480)
        selection_summary = " ".join(summary.split()[:28])

        reference = gdata.get("reference")
        primary_file = next(iter(file_names), None)
        if not reference and primary_file:
            pf_stem = primary_file.rsplit("/",1)[-1].split(".")[0].lower()
            for r in references:
                rs = r.rsplit("/",1)[-1].split(".")[0].lower()
                if rs == pf_stem:
                    reference = r
                    break
        alts = _url_variants(domain_url, reference, primary_file)
        resume_url = ""
        # prefer first alt that looks like doc
        for a in alts:
            if re.search(r"\.(pdf|docx?|rtf)\b", a, re.I):
                resume_url = a
                break
        if not resume_url and alts:
            resume_url = alts[0]

        cand = Candidate(
            candidate_id=gid,
            group_ids=[gid],
            name=name,
            name_conf=init_conf,
            name_source=init_source,
            name_debug=init_debug if CONFIG["INCLUDE_NAME_DEBUG"] else {},
            chunks=matched,
            file_names=list(file_names),
            reference_file=reference,
            raw_score_sum=total_score,
            summary=summary,
            selection_summary=selection_summary,
            aggregated_text=aggregated_text,
            emails=emails,
            phones=phones,
            resume_url=resume_url,
            resume_url_alts=alts
        )
        low = name.lower()
        cand.heading_like = _is_heading_like(low)
        cand.location_like = _is_location_line(low)
        cand.company_like = _looks_company_name(low)
        out.append(cand)
    return out

# --------------------- Duplicate Merge ---------------------
def _merge_duplicates(cands: List[Candidate]) -> List[Candidate]:
    email_map: Dict[str,Candidate] = {}
    phone_map: Dict[str,Candidate] = {}
    sig_map: Dict[str,Candidate] = {}
    merged: List[Candidate] = []

    def sig(c: Candidate):
        if c.emails:
            return "email:"+sorted(c.emails)[0]
        if c.phones:
            return "phone:"+sorted(c.phones)[0]
        return "sig:"+_canonical_signature(c.chunks)

    for c in cands:
        signature = sig(c)
        target = None
        for e in c.emails:
            if e in email_map:
                target = email_map[e]; break
        if not target:
            for p in c.phones:
                if p in phone_map:
                    target = phone_map[p]; break
        if not target and signature in sig_map:
            target = sig_map[signature]

        if target:
            target.group_ids.extend(c.group_ids)
            target.raw_score_sum += c.raw_score_sum
            target.chunks.extend(c.chunks)
            target.emails |= c.emails
            target.phones |= c.phones
            if (c.name_conf > target.name_conf and
                c.name.lower() not in STOP_HEADING_TOKENS and
                not c.name.lower().startswith("candidate")):
                target.name = c.name
                target.name_conf = c.name_conf
                target.name_source = "merged_better_name"
            # aggregated text extend
            if len(target.aggregated_text) < 4000:
                target.aggregated_text = (target.aggregated_text + "\n\n" + c.aggregated_text).strip()[:4000]
            if not target.resume_url and c.resume_url:
                target.resume_url = c.resume_url
                target.resume_url_alts = c.resume_url_alts
        else:
            merged.append(c)
            for e in c.emails: email_map[e] = c
            for p in c.phones: phone_map[p] = c
            sig_map[signature] = c
    return merged

# --------------------- Ranking ---------------------
def _normalize(cands: List[Candidate]):
    if not cands: return
    max_raw = max(c.raw_score_sum for c in cands) or 1.0
    for c in cands:
        c.raw_norm = c.raw_score_sum / max_raw

def _apply_keywords(cands: List[Candidate], keywords: List[str]):
    if not keywords:
        for c in cands: c.coverage_score = 0.0
        return
    for c in cands:
        combined = " ".join(ch["content"] for ch in c.chunks)
        c.coverage_score = _keyword_overlap(keywords, combined)

def _compute_bonuses(c: Candidate):
    b = 0.0
    if c.emails: b += CONFIG["BONUS_HAS_EMAIL"]
    if c.phones: b += CONFIG["BONUS_HAS_PHONE"]
    if len(c.chunks) > 1:
        steps = min(len(c.chunks)-1, int(CONFIG["BONUS_MULTI_CHUNK_MAX"]/CONFIG["BONUS_MULTI_CHUNK_STEP"]))
        b += steps * CONFIG["BONUS_MULTI_CHUNK_STEP"]
    if len(c.summary) > 250:
        b += CONFIG["BONUS_LONG_SUMMARY"]
    if c.heading_like:
        b += CONFIG["PENALTY_HEADING_NAME"]
        if len(c.chunks) == 1:
            b += CONFIG["PENALTY_SINGLE_CHUNK_HEADING"]
    if c.location_like:
        b += CONFIG["PENALTY_LOCATION_NAME"]
    if c.company_like:
        b += CONFIG["PENALTY_COMPANY_NAME"]
    c.bonus_total = b

def _composite(cands: List[Candidate]):
    for c in cands:
        _compute_bonuses(c)
        c.composite = (
            CONFIG["WEIGHT_RAW_SCORE"] * c.raw_norm +
            CONFIG["WEIGHT_KEYWORD_COVERAGE"] * c.coverage_score +
            CONFIG["WEIGHT_BONUSES"] * c.bonus_total
        )

def _prune(cands: List[Candidate]):
    if not cands: return
    scores = [c.raw_score_sum for c in cands]
    med = median(scores) if scores else 0
    thresh = med * CONFIG["PRUNE_SCORE_BELOW_FRAC_MEDIAN"]
    for c in cands:
        if (c.raw_score_sum < thresh and not c.emails and not c.phones and
            len(c.chunks) < CONFIG["PRUNE_REQUIRE_MIN_CHUNKS_IF_NO_CONTACT"]):
            c.pruned = True
            c.prune_reason = "low_score_no_contact"

def _rank_and_select(cands: List[Candidate], top_k: int) -> List[Candidate]:
    cands.sort(key=lambda x: x.composite, reverse=True)
    rank = 1
    for c in cands:
        if not c.pruned and rank <= top_k:
            c.final_rank = rank
            rank += 1
    selected = [c for c in cands if c.final_rank is not None]
    if len(selected) < top_k:
        for c in cands:
            if c.final_rank is None:
                selected.append(c)
                if len(selected) >= top_k: break
    return selected[:top_k]

# --------------------- MAIN ---------------------
def content_retrieval(
    job_description: str,
    email: str,
    domain_url: str = Config.CONTENT_DOMAIN_URL,
    api_endpoint: str = Config.CONTENT_RETRIEVAL_ENDPOINT,
    top_k: int = 5,
    ranking_mode: str = "deterministic",
    force_refresh_cache: bool = False,
) -> Dict[str, Any]:
    start = datetime.datetime.utcnow()
    if top_k > CONFIG["MAX_TOP_K"]:
        top_k = CONFIG["MAX_TOP_K"]

    cache_key = None
    if CONFIG["ENABLE_TTL_CACHE"]:
        cache_key = f"v18.1|{email}|{_hash_jd(job_description)}|{top_k}"
        if not force_refresh_cache and _global_cache:
            cached = _global_cache.get(cache_key)
            if cached:
                cached["diagnostics"]["cache_hit"] = True
                cached["retrieval_start"] = start.isoformat()
                cached["retrieval_end"] = datetime.datetime.utcnow().isoformat()
                return cached

    endpoint_url = f"{domain_url}{api_endpoint}"
    retrieved = None
    retrieval_error = None
    try:
        resp = send_request(
            endpoint_url,
            data={"email": email, "query": job_description},
            content_type="JSON",
            request_type="POST",
            total_retry=3,
        )
        if resp and resp.status_code == 200:
            try:
                retrieved = json.loads(resp.text)
            except Exception as e:
                retrieval_error = f"json_parse_error:{e}"
        else:
            retrieval_error = f"status_{getattr(resp,'status_code','NO_RESP')}"
    except Exception as e:
        retrieval_error = f"request_error:{e}"

    if not retrieved or "chunks" not in retrieved:
        end = datetime.datetime.utcnow()
        failure = {
            "retrieval_start": start.isoformat(),
            "retrieval_end": end.isoformat(),
            "top_candidates": [],
            "diagnostics": {
                "error": retrieval_error or "no_chunks",
                "raw_chunk_count": len(retrieved.get("chunks", [])) if retrieved else 0,
            },
        }
        if cache_key and _global_cache:
            _global_cache.set(cache_key, failure)
        return failure

    raw_chunks: List[Dict] = retrieved.get("chunks", [])
    references: List[str] = retrieved.get("reference", []) or []

    logger.info("RETRIEVAL v18.1: raw_chunks=%d references=%d", len(raw_chunks), len(references))

    groups = _det_group(raw_chunks, references)
    if len(groups) > CONFIG["MAX_GROUPS_PROCESS"]:
        logger.warning("Capping groups %d -> %d", len(groups), CONFIG["MAX_GROUPS_PROCESS"])
        groups = dict(list(groups.items())[:CONFIG["MAX_GROUPS_PROCESS"]])

    candidates = _build_candidates(groups, references, domain_url)
    before_merge = len(candidates)
    candidates = _merge_duplicates(candidates)
    after_merge = len(candidates)

    _normalize(candidates)
    jd_keywords = _extract_keywords(job_description)
    _apply_keywords(candidates, jd_keywords)
    _composite(candidates)
    _prune(candidates)

    selected = _rank_and_select(candidates, top_k)

    end = datetime.datetime.utcnow()
    diagnostics = {
        "raw_chunk_count": len(raw_chunks),
        "group_count_initial": len(groups),
        "candidates_before_merge": before_merge,
        "candidates_after_merge": after_merge,
        "pruned_total": sum(1 for c in candidates if c.pruned),
        "returned": len(selected),
        "jd_keyword_count": len(jd_keywords),
    }

    response = {
        "retrieval_start": start.isoformat(),
        "retrieval_end": end.isoformat(),
        "top_candidates": [c.to_json(CONFIG["INCLUDE_NAME_DEBUG"]) for c in selected],
        "diagnostics": diagnostics,
    }

    if cache_key and _global_cache:
        try:
            _global_cache.set(cache_key, response)
        except Exception:
            pass
    return response
