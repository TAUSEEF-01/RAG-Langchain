"""Gemini LLM implementation of BaseLLM using LangChain's ChatGoogleGenerativeAI.

Mirrors CohereLLM so the rest of the code can remain provider agnostic.
"""

from typing import Union, Iterator
from langchain_google_genai import ChatGoogleGenerativeAI

from src.llms.base_llm import BaseLLM


class GeminiLLM(BaseLLM):
    """Google Gemini chat model wrapper.

    Args:
        google_api_key (str): API key.
        model_name (str): Chat model name, default "gemini-1.5-flash" for speed/cost.
    """

    def __init__(self, google_api_key: str, model_name: str = "gemini-1.5-flash"):
        self.chat_model = ChatGoogleGenerativeAI(
            model=model_name, google_api_key=google_api_key
        )

    def get_llm(self):  # pragma: no cover - simple accessor
        return self.chat_model

    def generate(self, prompt: str, stream: bool = False) -> Union[str, Iterator[str]]:
        if stream:
            return self.chat_model.stream(prompt)
        return self.chat_model.invoke(prompt).content
