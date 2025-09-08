import asyncio
import datetime
import random
import time
from typing import Tuple, Optional

from openai import (
    OpenAI,
    AsyncOpenAI,
    RateLimitError,
    APITimeoutError,
    InternalServerError,
)
from django_core.config import Config


async def make_openai_request(
    prompt_message: str,
    model: str = None,
    temperature: float = 0.0,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    max_retries: int = 5,
) -> Tuple[Optional[dict], str, int]:
    """
    Make an OpenAI chat completion request with exponential backoff.

    Returns:
      (response_object, exception_log, retries_used)
      response_object: raw API response (dict-like)
      exception_log: concatenated string of retry exception info
      retries_used: number of retries actually performed
    """
    api_key = Config.OPEN_AI_KEY
    if not api_key:
        return None, "OPENAI_API_KEY not configured.", 0

    async_client = AsyncOpenAI(api_key=api_key)
    use_model = model or Config.GPT_3_MODEL or "gpt-3.5-turbo"

    exception_string = ""
    retries = 0
    delay = initial_delay

    while retries < max_retries:
        attempt_time = datetime.datetime.utcnow()
        try:
            response = await async_client.chat.completions.create(
                model=use_model,
                messages=[{"role": "user", "content": prompt_message}],
                temperature=temperature,
            )
            return response, exception_string, retries
        except (RateLimitError, APITimeoutError, InternalServerError) as e:
            e_time = datetime.datetime.utcnow()
            exception_string += f"{type(e).__name__}: {e} \t{(e_time - attempt_time).total_seconds():.2f}s\n"
            # Exponential backoff with optional jitter
            delay *= exponential_base * (1 + (random.random() if jitter else 0))
            await asyncio.sleep(delay)
            retries += 1
        except Exception as e:
            e_time = datetime.datetime.utcnow()
            exception_string += f"{type(e).__name__}: {e} \t{(e_time - attempt_time).total_seconds():.2f}s\n"
            return None, exception_string, retries

    exception_string += f"Max retries reached ({max_retries})."
    return None, exception_string, retries
