import os
import google.generativeai as genai


def estimate_tokens(text: str):
    """
    Estimate the number of tokens in a given text string using the tokenizer from Gemini.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.count_tokens(text)
            return response.total_tokens
        except Exception as e:
            print(f"Error with Gemini API: {e}")
    # Fallback heuristic: ~4 chars per token
    return max(1, len(text) // 4)


if __name__ == "__main__":
    test_text = "Hello, how are you?"
    token_count = estimate_tokens(test_text)
    print(f"Token count for '{test_text}': {token_count}")

    # Test with a longer text
    longer_text = "This is a longer piece of text to test the tokenization functionality with the Gemini API."
    longer_token_count = estimate_tokens(longer_text)
    print(f"Token count for longer text: {longer_token_count}")
