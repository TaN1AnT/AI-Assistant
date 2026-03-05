from langchain_google_vertexai import ChatVertexAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from app.config import settings, credentials

# Allow ~180 requests per minute (3 per second)
_rate_limiter = InMemoryRateLimiter(requests_per_second=3, check_every_n_seconds=0.1, max_bucket_size=10)

def get_gemini_llm(temperature: float = 0):
    """
    Returns a configured Gemini instance via Vertex AI.
    Uses the service account credentials loaded by config.py / vertexai.init().
    """
    return ChatVertexAI(
        model=settings.GEMINI_MODEL,
        temperature=temperature,
        max_retries=3,
        rate_limiter=_rate_limiter,
        project=settings.GOOGLE_PROJECT_ID,
        location=settings.GOOGLE_LOCATION,
        credentials=credentials,
    )
