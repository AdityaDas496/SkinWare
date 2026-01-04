# azure_language.py
import os
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

LANG_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
LANG_KEY = os.getenv("AZURE_LANGUAGE_KEY")

if not LANG_ENDPOINT or not LANG_KEY:
    raise RuntimeError("Azure AI Language not configured")

client = TextAnalyticsClient(
    endpoint=LANG_ENDPOINT,
    credential=AzureKeyCredential(LANG_KEY)
)

def analyze_model_output(result: dict) -> dict:
    """
    Uses Azure AI Language to analyze structured model output.
    """
    text = (
        f"dryness: {result['attributes']['dryness']}, "
        f"texture: {result['attributes']['texture']}, "
        f"redness: {result['attributes']['redness']}, "
        f"pigmentation: {result['attributes']['pigmentation']}, "
        f"concern_level: {result['concern_level']}"
    )

    response = client.analyze_sentiment([text])[0]

    return {
        "sentiment": response.sentiment,
        "confidence_scores": response.confidence_scores,
    }
