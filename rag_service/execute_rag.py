import os
import asyncio
import datetime
import logging
import requests

from api.helpers import cosine_similarity, generate_embedding_sync
from api.question_generator import generate_questions

from generation.generate_response import generate_query_response
from rag_service.utils import (
    fetch_source_from_reranked_chunks,
    post_process_rag_pipeline,
)
from rephrasing.rephrase import rephrase_query
from reranking.rerank import rerank_query
from reranking.utils import prepare_reranked_chunks_to_insert
from retrieval.content_retrieval import content_retrieval
from retrieval.utils import prepare_retrieved_chunks_to_insert

logger = logging.getLogger(__name__)

# ----------------- Farmstack Candidate Fetch (Configurable) -----------------
# Set this in your .env (or environment):
#   FARMSTACK_CANDIDATES_API="https://<your-farmstack>/be/api/candidates"
# The code will skip candidate logic if this is not provided.
FARMSTACK_CANDIDATES_API = os.getenv("FARMSTACK_CANDIDATES_API", "").strip()


def get_all_candidates_embeddings():
    """
    Fetch all candidates from Farmstack with precomputed embeddings.

    Expected response format (example):
    [
        {"name": "John Doe", "level": "Intermediate", "embedding": [...]},
        ...
    ]

    If FARMSTACK_CANDIDATES_API is not configured or request fails, returns [].
    """
    if not FARMSTACK_CANDIDATES_API:
        logger.info("FARMSTACK_CANDIDATES_API not set; skipping candidate fetch.")
        return []

    try:
        response = requests.get(FARMSTACK_CANDIDATES_API, timeout=15)
        response.raise_for_status()
        candidates = response.json()
        cleaned = []
        for c in candidates if isinstance(candidates, list) else []:
            cleaned.append({
                "name": c.get("name"),
                "level": c.get("level", "Unknown"),
                "embedding": c.get("embedding", []),
            })
        return cleaned
    except Exception as e:
        logger.error(f"Error fetching candidates from Farmstack: {e}", exc_info=True)
        return []


def execute_rag_pipeline(
    original_query,
    input_language_detected,
    email_id,
    user_name=None,
    message_id=None,
    chat_history=None,
):
    """
    Execute RAG pipeline to process rephrasing, retrieval, reranking, and response
    for the given query based on the available content.

    Additionally, when FARMSTACK_CANDIDATES_API is configured, it will:
      - compute an embedding for the original query,
      - score candidates via cosine similarity,
      - pick the top candidate,
      - generate interview questions, and
      - include them in response_map["candidate_questions"].

    This is non-blocking; if candidate fetch fails or config is missing, it simply skips.
    """
    generated_final_response = None
    retrieved_chunks = []
    response_map = {"message_id": message_id}
    message_data_to_insert_or_update = {"message_id": message_id}

    try:
        message_data_to_insert_or_update["main_bot_logic_start_time"] = datetime.datetime.now()

        # Step 1: Rephrase query
        rephrased_query_response = asyncio.run(
            rephrase_query(original_query, chat_history)
        )
        rephrased_query = rephrased_query_response.get("rephrased_query")

        # Step 2: Content retrieval
        retrieval_results = content_retrieval(rephrased_query, email_id)
        retrieved_chunks = (
            retrieval_results.get("retrieved_chunks", {}).get("chunks", [])
            if isinstance(retrieval_results, dict)
            else []
        )
        retrieved_chunk_data_to_insert = prepare_retrieved_chunks_to_insert(
            retrieved_chunks, message_id
        )

        # Step 3: Rerank
        reranked_query_response = asyncio.run(
            rerank_query(
                original_query,
                rephrased_query,
                email_id,
                retrieved_chunks,
            )
        )
        reranked_chunk_data_to_insert = prepare_reranked_chunks_to_insert(
            reranked_query_response.get("reranked_chunks", []), message_id
        )
        context_chunks = reranked_query_response.get("context_chunks")
        content_source = fetch_source_from_reranked_chunks(
            reranked_query_response.get("reranked_chunks")
        )

        # ----------------- Talent Acquisition (FS/FC tweak) -----------------
        # This is optional; if FARMSTACK_CANDIDATES_API is not set or fails, we skip cleanly.
        try:
            candidates = get_all_candidates_embeddings()
            if candidates:
                query_embedding = generate_embedding_sync(original_query)

                scored_candidates = []
                for candidate in candidates:
                    # Guard against missing/ill-formed embeddings
                    cand_emb = candidate.get("embedding") or []
                    score = cosine_similarity(query_embedding, cand_emb) if cand_emb else 0.0
                    scored_candidates.append({
                        "name": candidate.get("name"),
                        "level": candidate.get("level"),
                        "score": score,
                    })

                # Keep top-5 for UI, pick top-1 for question generation
                top_candidates = sorted(
                    scored_candidates, key=lambda x: x["score"], reverse=True
                )[:5]

                if top_candidates:
                    selected_candidate = top_candidates[0]
                    # Generate interview questions for the selected top candidate
                    # (If your generator needs embeddings, extend it to accept them.)
                    candidate_questions = generate_questions(
                        candidate_name=selected_candidate["name"],
                        candidate_level=selected_candidate["level"],
                        candidate_embeddings=None,  # or pass the real embedding if your generator uses it
                    )

                    response_map["top_candidates"] = top_candidates
                    response_map["selected_candidate"] = selected_candidate
                    response_map["candidate_questions"] = candidate_questions
        except Exception as talent_err:
            # Never break main flow
            logger.error(f"Talent engine step failed: {talent_err}", exc_info=True)

        # Step 4: Generate final response / answer for the query
        generated_response = asyncio.run(
            generate_query_response(
                original_query, user_name, context_chunks, rephrased_query
            )
        )
        generated_final_response = generated_response.get("response")

        # Timing / metadata
        now = datetime.datetime.now()
        message_data_to_insert_or_update["main_bot_logic_end_time"] = now
        message_data_to_insert_or_update["message_response_time"] = now
        message_data_to_insert_or_update["retrieved_chunks"] = str(context_chunks)
        message_data_to_insert_or_update["condensed_question"] = rephrased_query

        # RAG logging (post-process)
        post_process_rag_pipeline(
            message_id,
            rephrased_query_response,
            retrieved_chunk_data_to_insert,
            reranked_chunk_data_to_insert,
            reranked_query_response,
            generated_response,
        )

        # Final map
        response_map.update(
            {
                "message_id": message_id,
                "generated_final_response": generated_final_response,
                "source": content_source,
            }
        )

    except Exception as error:
        logger.error(error, exc_info=True)

    return response_map, message_data_to_insert_or_update

