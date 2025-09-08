from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, math

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def rerank(self, query: str, candidates: List[Dict], top_k=10):
        pairs = []
        meta = []
        for c in candidates[:top_k]:
            # Use concatenation of top chunk texts truncated
            txt = " ".join(ch["chunk_id"] + ": " + ch.get("text","")[:400] for ch in c.get("top_chunks",[]))
            pairs.append((query, txt[:1000]))
            meta.append(c)
        if not pairs:
            return candidates
        enc = self.tokenizer([p[0] for p in pairs],[p[1] for p in pairs],
                             truncation=True,padding=True,return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**enc).logits.squeeze(-1)
            scores = out.cpu().tolist()
        # Sigmoid to get 0..1
        import math
        probs = [1/(1+math.exp(-s)) for s in scores]
        for m,p in zip(meta, probs):
            m["rerank_prob"] = p
            m["final_score"] = 0.6*p + 0.4*m["resume_score"]
        return sorted(meta, key=lambda x: x["final_score"], reverse=True)
