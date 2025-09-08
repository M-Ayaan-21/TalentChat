import math, json, os, re
from collections import Counter, defaultdict
from typing import List, Dict, Any

STOP = {"a","an","the","to","of","in","on","and","or","for","with","is","are","as","at","by","from"}

TOKEN_RE = re.compile(r"[A-Za-z0-9+#]+")
SKILL_WHITELIST = {
    "python","java","kotlin","c++","c#","aws","gcp","azure","spark","hadoop","kafka",
    "sql","postgres","mysql","mongodb","react","node","unreal","unity","gameplay",
    "docker","kubernetes","airflow","etl","ml","ai","nlp","flutter","django","spring"
}

def tokenize(text: str):
    for t in TOKEN_RE.findall(text.lower()):
        if t in STOP:
            continue
        yield t

class ChunkIndex:
    def __init__(self):
        self.chunks: List[Dict[str,Any]] = []
        self.doc_freq = Counter()
        self.N = 0
        self.idf = {}
        self.resume_chunk_map = defaultdict(list)

    def add_chunk(self, chunk_id: str, resume_id: str, text: str):
        tokens = list(tokenize(text))
        tf = Counter(tokens)
        self.chunks.append({
            "chunk_id": chunk_id,
            "resume_id": resume_id,
            "text": text,
            "tf": tf,
            "tokens": tokens
        })
        self.N += 1
        for token in set(tokens):
            self.doc_freq[token] += 1
        self.resume_chunk_map[resume_id].append(chunk_id)

    def finalize(self, min_df=1, max_df_ratio=0.85):
        self.idf = {}
        for term, df in self.doc_freq.items():
            if df < min_df or df > max_df_ratio*self.N:
                continue
            # Skill boost
            boost = 1.25 if term in SKILL_WHITELIST else 1.0
            self.idf[term] = boost * math.log(1 + (self.N - df + 0.5)/(df + 0.5))

    def bm25(self, tf: int, dl: int, avgdl: float, idf: float, k1=1.4, b=0.75):
        return idf * ((tf*(k1+1)) / (tf + k1*(1 - b + b*(dl/avgdl))))

    def lexical_scores(self, query_tokens: List[str]):
        scores = []
        avgdl = sum(len(c["tokens"]) for c in self.chunks)/max(1,len(self.chunks))
        qt = [t for t in query_tokens if t in self.idf]
        qt_set = set(qt)
        for c in self.chunks:
            dl = len(c["tokens"])
            s = 0.0
            matched = []
            for t in qt_set:
                if t in c["tf"]:
                    s += self.bm25(c["tf"][t], dl, avgdl, self.idf[t])
                    matched.append(t)
            if s > 0:
                scores.append({
                    "chunk_id": c["chunk_id"],
                    "resume_id": c["resume_id"],
                    "lex_score": s,
                    "matched": matched
                })
        return scores
