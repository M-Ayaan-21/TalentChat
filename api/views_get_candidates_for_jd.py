from retrieval.llm_rerank_integration import apply_llm_rerank

def get_candidates_for_jd(request):
    jd_text = (request.data.get("jd") or request.data.get("job_description") or "").strip()

    # Your existing retrieval -> must produce initial_candidates with resume_id + chunks
    initial_candidates = build_initial_candidates(jd_text)

    # Optional debug to confirm wiring
    print("DEBUG_BEFORE_LLM", {"pool": len(initial_candidates)})

    # LLM rerank (with safe fallback)
    try:
        final_candidates = apply_llm_rerank(jd_text, initial_candidates, top_k=5)
    except Exception as e:
        print("LLM_RERANK_ERROR", str(e))
        final_candidates = initial_candidates[:5]

    return JsonResponse({"results": final_candidates}, safe=False)
