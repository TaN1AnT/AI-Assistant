from langchain_google_vertexai import ChatVertexAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from vertexai.generative_models import HarmCategory, HarmBlockThreshold
from app.config import settings, credentials

# Allow ~60 requests per minute (1 per second)
_rate_limiter = InMemoryRateLimiter(requests_per_minute=180, check_every_n_seconds=0.1, max_bucket_size=10)

def get_gemini_llm(temperature: float = 0):
    """
    Returns a configured Gemini Flash instance with retry logic.
    Uses explicit service account credentials from config.
    """
    kwargs = dict(
        model=settings.GEMINI_MODEL,
        project=settings.GOOGLE_PROJECT_ID,
        location=settings.GOOGLE_LOCATION,
        temperature=temperature,
        max_retries=3,
        rate_limiter=_rate_limiter,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        },
    )
    if credentials:
        kwargs["credentials"] = credentials
    return ChatVertexAI(**kwargs)