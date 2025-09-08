import numpy as np, math
from typing import List, Dict, Any, Tuple
from .index_builder import tokenize

class HybridRetriever:
    def __init__(self, chunk_index, embedder):
        self.index = chunk_index
        self.embedder = embedder  # must provide encode(list_of_texts) -> np.array normalized

    def expand_query(self, q: str) -> Dict[str,Any]:
        base = list(tokenize(q))
        # naive expansion example: add unity/unreal for 'game'
        expanded = set(base)
        if "game" in base:
            expanded.update(["unity","unreal","gameplay"])
        if "data" in base and "engineer" in base:
            expanded.update(["airflow","etl","pipeline","spark","kafka"])
        return {"tokens": list(expanded), "original": base}

    def semantic_chunk_scores(self, chunks_subset: List[Dict[str,Any]], query_text: str):
        texts = [c["text"][:1000] for c in chunks_subset]
        q_vec = self.embedder.encode([query_text])[0]
        c_vecs = self.embedder.encode(texts)
        sims = (c_vecs @ q_vec)  # assuming normalized
        return sims

    def retrieve(self, query: str, top_k=40):
        qinfo = self.expand_query(query)
        lex = self.index.lexical_scores(qinfo["tokens"])
        if not lex:
            # fallback: take ALL chunks for semantic
            subset_chunks = self.index.chunks
        else:
            # keep broader set for semantic (more than top_k to allow rescoring)
            lex_sorted = sorted(lex, key=lambda x: x["lex_score"], reverse=True)
            top_for_sem = lex_sorted[:max(top_k*3, 60)]
            subset_ids = {x["chunk_id"] for x in top_for_sem}
            subset_chunks = [c for c in self.index.chunks if c["chunk_id"] in subset_ids]

        # Build map chunk_id -> chunk dict
        chunk_map = {c["chunk_id"]: c for c in self.index.chunks}
        sem_scores = self.semantic_chunk_scores(subset_chunks, query)
        sem_map = {c["chunk_id"]: float(s) for c,s in zip(subset_chunks, sem_scores)}

        # Normalize lexical
        if lex:
            max_lex = max(x["lex_score"] for x in lex)
            for x in lex:
                x["lex_norm"] = x["lex_score"]/max_lex
        else:
            lex = []
        # Build merged chunk list (union)
        merged = {}
        for x in lex:
            merged[x["chunk_id"]] = {
                "chunk_id": x["chunk_id"],
                "resume_id": x["resume_id"],
                "lex_norm": x.get("lex_norm",0.0),
                "matched": x["matched"],
                "sem": sem_map.get(x["chunk_id"], 0.0)
            }
        for cid, s in sem_map.items():
            if cid not in merged:
                merged[cid] = {
                    "chunk_id": cid,
                    "resume_id": chunk_map[cid]["resume_id"],
                    "lex_norm": 0.0,
                    "matched": [],
                    "sem": s
                }

        # Normalize semantic
        sem_vals = [v["sem"] for v in merged.values()]
        if sem_vals:
            min_sem, max_sem = min(sem_vals), max(sem_vals)
            rng = max_sem - min_sem + 1e-6
            for v in merged.values():
                v["sem_norm"] = (v["sem"] - min_sem)/rng
        else:
            for v in merged.values():
                v["sem_norm"] = 0.0

        # Dynamic weighting
        q_len = len(qinfo["tokens"])
        if q_len < 3:
            w_sem, w_lex = 0.60, 0.30
        elif q_len > 10:
            w_sem, w_lex = 0.35, 0.55
        else:
            w_sem, w_lex = 0.45, 0.45

        for v in merged.values():
            distinct_match = len(set(v["matched"]))
            coverage = distinct_match / max(1,q_len)
            v["hybrid_score"] = (
                w_lex * v["lex_norm"] +
                w_sem * v["sem_norm"] +
                0.10 * coverage
            )

        # Aggregate to resume level
        per_resume = {}
        for v in merged.values():
            r = per_resume.setdefault(v["resume_id"], {"chunks":[]})
            r["chunks"].append(v)

        results = []
        for rid, data in per_resume.items():
            chunks_sorted = sorted(data["chunks"], key=lambda x: x["hybrid_score"], reverse=True)
            top_scores = [c["hybrid_score"] for c in chunks_sorted[:5]]
            max_chunk = top_scores[0]
            mean_top3 = sum(top_scores[:3])/max(1,len(top_scores[:3]))
            coverage_tokens = set()
            for c in chunks_sorted[:3]:
                coverage_tokens.update(c["matched"])
            coverage_bonus = len(coverage_tokens)/max(1,q_len) * 0.05
            resume_score = max_chunk + 0.5*mean_top3 + coverage_bonus
            results.append({
                "resume_id": rid,
                "resume_score": resume_score,
                "max_chunk": max_chunk,
                "mean_top3": mean_top3,
                "coverage_bonus": coverage_bonus,
                "top_chunks": chunks_sorted[:3]
            })

        # Thresholding
        if results:
            max_r = max(r["resume_score"] for r in results)
            # Drop very weak
            filtered = [
                r for r in results
                if r["resume_score"] >= 0.35*max_r and r["resume_score"] >= 0.40
            ]
        else:
            filtered = []

        final_sorted = sorted(filtered, key=lambda x: x["resume_score"], reverse=True)[:top_k]

        diagnostics = {
            "query_tokens": qinfo["tokens"],
            "lex_candidates": len(lex),
            "sem_subset": len(subset_chunks),
            "merged_chunks": len(merged),
            "resume_candidates": len(results),
            "returned": len(final_sorted)
        }

        return final_sorted, diagnostics
