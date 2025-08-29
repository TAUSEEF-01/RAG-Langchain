"""
Wrapper for Google Gemini embedding models using LangChain's GoogleGenerativeAIEmbeddings.

This mirrors the CohereEmbedding / HFEmbedding helpers already present so the rest of the
application can obtain a unified embeddings object via the factory.

Usage:
    gem = GeminiEmbedding(google_api_key="...", model_name="models/text-embedding-004")
    embeddings = gem.get_embeddings()
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings


class GeminiEmbedding:
    """Utility initializer for Gemini embedding models.

    Args:
        google_api_key (str): Google Generative AI (Gemini) API key.
        model_name (str): Embedding model name. Default: "models/text-embedding-004".
    """

    def __init__(
        self, google_api_key: str, model_name: str = "models/text-embedding-004"
    ) -> None:
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name, google_api_key=google_api_key
        )

    def get_embeddings(self):  # pragma: no cover - simple accessor
        """Return underlying embeddings object."""
        return self.embeddings
