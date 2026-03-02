from langchain_google_vertexai import ChatVertexAI
from vertexai.generative_models import HarmCategory, HarmBlockThreshold
from app.config import settings, credentials

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