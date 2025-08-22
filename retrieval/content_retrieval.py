import datetime
import json
import logging
from collections import defaultdict

from common.utils import send_request
from django_core.config import Config

logger = logging.getLogger(__name__)


def content_retrieval(
    job_description: str,
    email: str,
    domain_url=Config.CONTENT_DOMAIN_URL,
    api_endpoint=Config.CONTENT_RETRIEVAL_ENDPOINT,
    top_k: int = 5,
):
    """
    Retrieve top K candidate resumes that best match the given job description.

    Args:
        job_description (str): The input job description text.
        email (str): The user email (used for authentication/tracking in FS).
        domain_url (str): Base domain of FS.
        api_endpoint (str): Retrieval endpoint of FS.
        top_k (int): Number of best candidates to return.

    Returns:
        dict: {
            "retrieval_start": datetime,
            "retrieval_end": datetime,
            "top_candidates": [
                {
                    "candidate_id": str,
                    "candidate_name": str,
                    "resume_title": str,
                    "score": float,
                    "matched_chunks": list
                },
                ...
            ]
        }
    """

    response_map = {}
    retrieval_start = datetime.datetime.now()
    retrieval_end = None

    content_retrieval_url = f"{domain_url}{api_endpoint}"
    retrieved_content = None

    try:
        response = send_request(
            content_retrieval_url,
            data={"email": email, "query": job_description},
            content_type="JSON",
            request_type="POST",
            total_retry=3,
        )

        if response and response.status_code == 200:
            retrieved_content = json.loads(response.text)
        else:
            logger.error("FS retrieval failed: %s", response.text if response else "No response")
            retrieved_content = None

    except Exception as error:
        logger.error("Error during FS retrieval", exc_info=True)
        retrieved_content = None

    retrieval_end = datetime.datetime.now()

    if not retrieved_content or "data" not in retrieved_content:
        response_map.update(
            {
                "retrieval_start": retrieval_start,
                "retrieval_end": retrieval_end,
                "top_candidates": [],
            }
        )
        return response_map

    # Group chunks by candidate
    candidate_scores = defaultdict(lambda: {"score": 0.0, "chunks": [], "resume_title": "", "candidate_name": ""})

    for chunk in retrieved_content["data"]:
        candidate_id = chunk.get("content_id") or chunk.get("id")
        candidate_name = chunk.get("file_name", "Unknown")
        resume_title = chunk.get("title", "Resume")
        score = chunk.get("score", 0.0)

        candidate_scores[candidate_id]["score"] += score
        candidate_scores[candidate_id]["chunks"].append(chunk)
        candidate_scores[candidate_id]["resume_title"] = resume_title
        candidate_scores[candidate_id]["candidate_name"] = candidate_name

    # Rank candidates
    ranked_candidates = sorted(
        [
            {
                "candidate_id": cid,
                "candidate_name": data["candidate_name"],
                "resume_title": data["resume_title"],
                "score": data["score"],
                "matched_chunks": data["chunks"],
            }
            for cid, data in candidate_scores.items()
        ],
        key=lambda x: x["score"],
        reverse=True,
    )[:top_k]

    response_map.update(
        {
            "retrieval_start": retrieval_start,
            "retrieval_end": retrieval_end,
            "top_candidates": ranked_candidates,
        }
    )

    return response_map

