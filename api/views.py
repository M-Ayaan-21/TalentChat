"""
api/views.py

Unified view layer including:
 - Candidate retrieval (stores enriched bundles for later question generation)
 - Interview question generation (JD + candidate resume context -> OpenAI or fallback)
 - Chat endpoints (text, voice, TTS, ASR)
 - Language preference endpoints
 - Index / health

Requires:
 - retrieval/content_retrieval.py (v18.1 or later) exposing content_retrieval
 - api/question_generator.py exposing generate_candidate_questions
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, List, Optional

from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from rest_framework import status
from rest_framework.decorators import action, api_view
from rest_framework.response import Response
from rest_framework.viewsets import GenericViewSet

from django.core.files.uploadedfile import InMemoryUploadedFile

# Project utilities
from api.utils import (
    authenticate_user_based_on_email,
    handle_input_query,
    process_input_audio_to_base64,
    process_output_audio,
    process_query,
    process_transcriptions,
)
from common.constants import Constants
from common.utils import get_user_by_email, set_user_preferred_language
from language_service.utils import get_all_languages, get_language_by_id

# Retrieval + Question Generation
from retrieval.content_retrieval import content_retrieval
from api.question_generator import generate_candidate_questions

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Constants / simple validators
# -------------------------------------------------------------------
ALLOWED_RANKING_MODES = {"deterministic", "llm"}  # currently deterministic emphasized
DEFAULT_TOP_K = 5
MAX_TOP_K = 15


def _validate_top_k(v: Any) -> int:
    try:
        k = int(v)
        return max(1, min(MAX_TOP_K, k))
    except Exception:
        return DEFAULT_TOP_K


def _validate_ranking_mode(m: Optional[str]) -> str:
    if not m:
        return "deterministic"
    m = m.lower()
    return m if m in ALLOWED_RANKING_MODES else "deterministic"


# -------------------------------------------------------------------
# Retrieval Cache
# Per email:
# {
#   "jd": <last job description>,
#   "candidates_map": {
#        candidate_id: {
#          candidate_id, candidate_name,
#          selection_summary, deterministic_summary,
#          aggregated_text, matched_chunks, resume_url
#        }
#    }
# }
# -------------------------------------------------------------------
LAST_RETRIEVAL_CACHE: Dict[str, Dict[str, Any]] = {}


# -------------------------------------------------------------------
# Basic pages
# -------------------------------------------------------------------
def index(request):
    return render(request, "index.html")


home = index  # alias


def health(request):
    return HttpResponse("ok", status=200)


# -------------------------------------------------------------------
# Candidate Retrieval Endpoint
# -------------------------------------------------------------------
@csrf_exempt
@api_view(["POST"])
def get_candidates_for_jd(request):
    """
    POST JSON:
    {
      "email_id": "...",
      "job_description": "...",
      "ranking_mode": "deterministic" | "llm",
      "top_k": 5
    }

    Returns:
        Retrieval payload: { top_candidates: [...], diagnostics: {...} }

    Side effects:
        Populates LAST_RETRIEVAL_CACHE[email_id] with enriched candidate bundles
        for later question generation.
    """
    data = request.data or {}
    email_id = data.get("email_id")
    jd = data.get("job_description")
    ranking_mode = _validate_ranking_mode(data.get("ranking_mode"))
    top_k = _validate_top_k(data.get("top_k"))

    if not email_id or not jd:
        return Response({"error": "Missing email_id or job_description"}, status=400)

    try:
        result = content_retrieval(
            job_description=jd,
            email=email_id,
            top_k=top_k,
            ranking_mode=ranking_mode,
        )

        bundle_map: Dict[str, Dict[str, Any]] = {}
        for c in result.get("top_candidates", []):
            bundle_map[c["candidate_id"]] = {
                "candidate_id": c["candidate_id"],
                "candidate_name": c.get("candidate_name"),
                "selection_summary": c.get("selection_summary"),
                "deterministic_summary": c.get("deterministic_summary"),
                "aggregated_text": c.get("aggregated_text", ""),
                "matched_chunks": c.get("matched_chunks", []),
                "resume_url": c.get("resume_url"),
            }

        LAST_RETRIEVAL_CACHE[email_id] = {
            "jd": jd,
            "candidates_map": bundle_map,
        }

        return Response(result, status=200)
    except Exception as e:
        logger.error("Candidate retrieval failed: %s", e, exc_info=True)
        return Response({"error": "retrieval_failed", "detail": str(e)}, status=500)


# -------------------------------------------------------------------
# Interview Question Generation
# -------------------------------------------------------------------
@csrf_exempt
@api_view(["POST"])
def generate_interview_questions(request):
    """
    POST JSON:
    {
      "candidate_id": "...",
      "candidate_name": "...",          (optional)
      "summary": "...",                 (optional – fallback if cache missing)
      "job_description": "...",         (optional – fallback to cached JD if email_id provided)
      "email_id": "..."                 (optional but recommended)
    }

    Returns: { "questions": [ ... ] }

    Workflow:
      - Retrieve candidate bundle from LAST_RETRIEVAL_CACHE[email_id] if available.
      - Use job_description from payload or cached JD.
      - Combine JD + aggregated_text + summaries and call generate_candidate_questions.
    """
    data = request.data or {}
    candidate_id = data.get("candidate_id")
    candidate_name = data.get("candidate_name") or "Candidate"
    supplied_summary = data.get("summary") or ""
    email_id = data.get("email_id")
    jd = data.get("job_description")

    if not candidate_id:
        return Response({"error": "candidate_id required"}, status=400)

    # Base bundle (will enrich if cache present)
    bundle = {
        "candidate_name": candidate_name,
        "selection_summary": supplied_summary,
        "deterministic_summary": supplied_summary,
        "aggregated_text": "",
        "matched_chunks": [],
        "resume_url": None,
    }

    if email_id and email_id in LAST_RETRIEVAL_CACHE:
        cache_entry = LAST_RETRIEVAL_CACHE[email_id]
        if not jd:
            jd = cache_entry.get("jd")
        cm = cache_entry.get("candidates_map", {})
        if candidate_id in cm:
            cached = cm[candidate_id]
            bundle.update(
                {
                    "candidate_name": cached.get("candidate_name") or candidate_name,
                    "selection_summary": cached.get("selection_summary")
                    or cached.get("deterministic_summary"),
                    "deterministic_summary": cached.get("deterministic_summary"),
                    "aggregated_text": cached.get("aggregated_text", ""),
                    "matched_chunks": cached.get("matched_chunks", []),
                    "resume_url": cached.get("resume_url"),
                }
            )

    if not jd:
        return Response(
            {
                "error": "job_description missing and no cached JD found for this email"
            },
            status=400,
        )

    try:
        questions = generate_candidate_questions(jd, bundle)
        return Response({"questions": questions}, status=200)
    except Exception as e:
        logger.error("Question generation failed: %s", e, exc_info=True)
        return Response(
            {"error": "question_generation_failed", "detail": str(e)}, status=500
        )


# -------------------------------------------------------------------
# Chat / Voice Q&A ViewSet
# -------------------------------------------------------------------
class ChatAPIViewSet(GenericViewSet):
    """
    Chat Service ViewSet
      - get_answer_for_text_query
      - synthesise_audio
      - transcribe_audio
      - get_answer_by_voice_query
    """

    authentication_classes: list = []

    @action(detail=False, methods=["post"])
    def get_answer_for_text_query(self, request):
        email_id = request.data.get("email_id")
        original_query = request.data.get("query")
        resp = Response({"message": None, "query": original_query, "error": False})

        try:
            user = authenticate_user_based_on_email(email_id)
            if not user:
                resp.data["message"] = "Invalid Email ID"
                resp.status_code = status.HTTP_401_UNAUTHORIZED
                return resp
            if not original_query:
                resp.data["message"] = "Please submit a query."
                resp.status_code = status.HTTP_400_BAD_REQUEST
                return resp

            result = process_query(original_query, email_id, user)
            resp.data.update(
                {
                    "message": "Successful retrieval of response for above query",
                    "message_id": result.get("message_id"),
                    "response": result.get("translated_response"),
                    "source": result.get("source"),
                    "follow_up_questions": result.get("follow_up_questions"),
                }
            )
        except Exception as e:
            logger.error("Text query failure", exc_info=True)
            resp.data.update(
                {"message": "Something went wrong", "error": True, "detail": str(e)}
            )
            resp.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return resp

    @action(detail=False, methods=["post"])
    def synthesise_audio(self, request):
        email_id = request.data.get("email_id")
        text = request.data.get("text")
        message_id = request.data.get("message_id")
        resp = Response({"message": None, "error": False, "audio": None})

        try:
            user = authenticate_user_based_on_email(email_id)
            if not user:
                resp.data["message"] = "Invalid Email ID"
                resp.status_code = status.HTTP_401_UNAUTHORIZED
                return resp
            if not text:
                resp.data["message"] = "Please submit text for audio synthesis."
                resp.status_code = status.HTTP_400_BAD_REQUEST
                return resp

            audio_b64 = process_output_audio(text, message_id)
            if not audio_b64:
                resp.data.update(
                    {
                        "message": "Unable to synthesize audio currently.",
                        "audio": None,
                    }
                )
                resp.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
                return resp

            resp.data.update(
                {"message": "Audio synthesis successful", "text": text, "audio": audio_b64}
            )
        except Exception as e:
            logger.error("TTS error", exc_info=True)
            resp.data.update(
                {"message": "Something went wrong", "error": True, "detail": str(e)}
            )
            resp.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return resp

    @action(detail=False, methods=["post"])
    def transcribe_audio(self, request):
        email_id = request.data.get("email_id")
        original_query = request.data.get("query")
        language_bcp = request.data.get(
            "query_language_bcp_code", Constants.LANGUAGE_BCP_CODE_NATIVE
        )

        resp = Response(
            {
                "message": None,
                "heard_input_query": None,
                "heard_input_audio": original_query,
                "confidence_score": 0,
                "error": False,
            }
        )

        try:
            user = authenticate_user_based_on_email(email_id)
            if not user:
                resp.data["message"] = "Invalid Email ID"
                resp.status_code = status.HTTP_401_UNAUTHORIZED
                return resp
            if not original_query and not request.FILES:
                resp.data["message"] = "Please provide audio (file or base64)."
                resp.status_code = status.HTTP_400_BAD_REQUEST
                return resp

            input_query = (
                request.FILES.get("query")
                if request.FILES and "query" in request.FILES
                else original_query
            )
            if isinstance(input_query, InMemoryUploadedFile):
                input_query.seek(0)
                file_bytes = input_query.read()
                input_query = base64.b64encode(file_bytes)

            input_query_file = handle_input_query(input_query)
            if not input_query_file:
                resp.data["message"] = "Invalid file or base64 audio."
                resp.status_code = status.HTTP_400_BAD_REQUEST
                return resp

            transcribed = process_transcriptions(
                input_query_file,
                email_id,
                user,
                language_bcp_code=language_bcp,
            )

            confidence = transcribed.get("confidence_score", 0)
            heard_text = transcribed.get("transcriptions")
            resp.data.update(
                {
                    "message": "Transcription complete."
                    if confidence > Constants.ASR_DEFAULT_CONFIDENCE_SCORE
                    else "Low confidence transcription.",
                    "message_id": transcribed.get("message_id"),
                    "confidence_score": confidence,
                    "heard_input_query": heard_text,
                }
            )

            if (
                confidence > Constants.ASR_DEFAULT_CONFIDENCE_SCORE
                and heard_text
            ):
                resp.data["heard_input_audio"] = process_input_audio_to_base64(
                    heard_text, transcribed.get("message_id")
                )

        except Exception as e:
            logger.error("ASR error", exc_info=True)
            resp.data.update(
                {"message": "Something went wrong", "error": True, "detail": str(e)}
            )
            resp.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return resp

    @action(detail=False, methods=["post"])
    def get_answer_by_voice_query(self, request):
        email_id = request.data.get("email_id")
        resp = Response(
            {
                "message": None,
                "heard_input_query": None,
                "heard_input_audio": None,
                "confidence_score": 0,
                "error": False,
            }
        )

        try:
            transcribed_resp = self.transcribe_audio(request)
            resp.data.update(transcribed_resp.data)
            resp.status_code = transcribed_resp.status_code

            if transcribed_resp.status_code != 200:
                return resp

            confidence = resp.data.get("confidence_score", 0)
            if confidence <= Constants.ASR_DEFAULT_CONFIDENCE_SCORE:
                resp.data["message"] = "Transcription confidence too low to proceed."
                return resp

            synthetic = request
            synthetic.data._mutable = True  # type: ignore
            synthetic.data["query"] = resp.data.get("heard_input_query")

            text_answer = self.get_answer_for_text_query(synthetic)
            if text_answer.status_code == 200:
                resp.data.update(
                    {
                        "response": text_answer.data.get("response"),
                        "follow_up_questions": text_answer.data.get(
                            "follow_up_questions"
                        ),
                        "message": "Voice query answered successfully",
                    }
                )
            else:
                resp.data["message"] = "Transcription succeeded; answer generation failed."
        except Exception as e:
            logger.error("Voice Q&A error", exc_info=True)
            resp.data.update(
                {"message": "Something went wrong", "error": True, "detail": str(e)}
            )
            resp.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return resp


# -------------------------------------------------------------------
# Language ViewSet
# -------------------------------------------------------------------
class LanguageViewSet(GenericViewSet):
    authentication_classes: list = []

    @action(detail=False, methods=["get"])
    def languages(self, request):
        email_id = request.GET.get("email_id")
        resp = Response({"message": None, "error": False, "language_data": []})
        try:
            user = authenticate_user_based_on_email(email_id)
            if not user:
                resp.data["message"] = "Invalid Email ID"
                resp.status_code = status.HTTP_401_UNAUTHORIZED
                return resp
            languages = get_all_languages()
            resp.data.update(
                {
                    "message": "Successful retrieval of supported language list."
                    if languages
                    else "No languages found.",
                    "language_data": languages,
                }
            )
        except Exception as e:
            logger.error("Language list error", exc_info=True)
            resp.data.update(
                {"message": "Something went wrong", "error": True, "detail": str(e)}
            )
            resp.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return resp

    @action(detail=False, methods=["post"])
    def set_language(self, request):
        email_id = request.data.get("email_id")
        language_id = request.data.get("language_id")
        resp = Response({"message": None, "error": False})

        try:
            user = authenticate_user_based_on_email(email_id)
            if not user:
                resp.data["message"] = "Invalid Email ID"
                resp.status_code = status.HTTP_401_UNAUTHORIZED
                return resp
            if not language_id:
                resp.data["message"] = "Language ID not submitted"
                resp.status_code = status.HTTP_400_BAD_REQUEST
                return resp

            language_id_int = int(language_id)
            language_dict = get_language_by_id(language_id_int)
            if not language_dict or language_dict.get("language_id") != language_id_int:
                resp.data["message"] = f"Language with ID {language_id} does not exist."
                resp.status_code = status.HTTP_400_BAD_REQUEST
                return resp

            db_user = get_user_by_email(email_id)
            user_id = db_user.get("user_id") if db_user else None
            saved = set_user_preferred_language(user_id, language_id_int)
            if saved:
                resp.data["message"] = (
                    f"Saved user's ({email_id}) preferred language: "
                    f"{language_dict.get('display_name')}"
                )
                resp.status_code = status.HTTP_200_OK
            else:
                resp.data["message"] = (
                    f"Unable to save user's ({email_id}) preferred language: "
                    f"{language_dict.get('display_name')}"
                )
                resp.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        except Exception as e:
            logger.error("Set language error", exc_info=True)
            resp.data.update(
                {"message": "Something went wrong", "error": True, "detail": str(e)}
            )
            resp.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return resp


__all__ = [
    "index",
    "home",
    "health",
    "get_candidates_for_jd",
    "generate_interview_questions",
    "ChatAPIViewSet",
    "LanguageViewSet",
]
