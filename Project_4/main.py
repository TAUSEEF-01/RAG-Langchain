import argparse
import os
from typing import List

import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    pass
# model = genai.GenerativeModel("gemini-pro")
model = None
MODEL_CANDIDATES = []
# Build candidate list (user override first)
_user_model = os.getenv("GENAI_MODEL")
if _user_model:
    MODEL_CANDIDATES.append(_user_model)
MODEL_CANDIDATES += [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.0-pro",
    "gemini-pro",  # legacy
]
# Deduplicate preserving order
_seen = set()
MODEL_CANDIDATES = [m for m in MODEL_CANDIDATES if not (m in _seen or _seen.add(m))]


def build_prompt(query: str, context: List[str]) -> str:
    """
    Builds a prompt for the LLM. #

    This function builds a prompt for the LLM. It takes the original query,
    and the returned context, and asks the model to answer the question based only
    on what's in the context, not what's in its weights.

    Args:
    query (str): The original query.
    context (List[str]): The context of the query, returned by embedding search.

    Returns:
    A prompt for the LLM (str).
    """

    base_prompt = {
        "content": "I am going to ask you a question, which I would like you to answer"
        " based only on the provided context, and not any other information."
        " If there is not enough information in the context to answer the question,"
        ' say "I am not sure", then try to make a guess.'
        " Break your answer up into nicely readable paragraphs.",
    }
    user_prompt = {
        "content": f" The question is '{query}'. Here is all the context you have:"
        f'{(" ").join(context)}',
    }

    # combine the prompts to output a single prompt string
    system = f"{base_prompt['content']} {user_prompt['content']}"

    return system


def _select_model():
    global model
    last_err = None
    for name in MODEL_CANDIDATES:
        try:
            model = genai.GenerativeModel(name)
            return name
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Failed to initialize any Gemini model from {MODEL_CANDIDATES}: {last_err}"
    )


def get_gemini_response(query: str, context: List[str]) -> str:
    global model
    if model is None:
        chosen = _select_model()
        print(f"Using Gemini model: {chosen}")
    prompt = build_prompt(query, context)
    try:
        response = model.generate_content(prompt)
        return response.text or "(No text returned)"
    except Exception as e:
        # If model not found, try next candidate automatically once
        if "not found" in str(e).lower():
            print(f"Model error ({e}). Trying next candidate...")
            # Remove current model from list and retry
            if MODEL_CANDIDATES and model:
                try:
                    current_name = getattr(model, "model_name", None)
                    if current_name and current_name in MODEL_CANDIDATES:
                        MODEL_CANDIDATES.remove(current_name)
                except Exception:
                    pass
            model = None
            return get_gemini_response(query, context)
        raise


def main(
    collection_name: str = "documents_collection", persist_directory: str = "."
) -> None:
    # Check if the GOOGLE_API_KEY environment variable is set. Prompt the user to set it if not.
    google_api_key = None
    if "GOOGLE_API_KEY" not in os.environ:
        gapikey = input("Please enter your Google API Key: ")
        genai.configure(api_key=gapikey)
        google_api_key = gapikey
    else:
        google_api_key = os.environ["GOOGLE_API_KEY"]
        genai.configure(api_key=google_api_key)

    # Instantiate a persistent chroma client in the persist_directory.
    # This will automatically load any previously saved collections.
    # Learn more at docs.trychroma.com
    # client = chromadb.PersistentClient(path=persist_directory)
    chroma_api_key = os.getenv("CHROMA_API_KEY")
    chroma_tenant = os.getenv("CHROMA_TENANT")
    chroma_database = os.getenv("CHROMA_DATABASE")

    missing = [
        name
        for name, val in [
            ("CHROMA_API_KEY", chroma_api_key),
            ("CHROMA_TENANT", chroma_tenant),
            ("CHROMA_DATABASE", chroma_database),
        ]
        if not val
    ]

    if missing:
        raise RuntimeError(
            "Missing required Chroma Cloud environment variables: "
            + ", ".join(missing)
            + ". Set them in your environment or .env file."
        )

    client = chromadb.CloudClient(
        api_key=chroma_api_key,
        tenant=chroma_tenant,
        database=chroma_database,
    )

    # create embedding function
    try:
        embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=google_api_key,
            model_name=os.getenv("EMBED_MODEL", "models/embedding-001"),
            task_type="RETRIEVAL_QUERY",
        )
    except AttributeError as e:
        raise RuntimeError(
            "Upgrade chromadb: missing GoogleGenerativeAiEmbeddingFunction"
        ) from e

    # Get the collection.
    collection = client.get_collection(
        name=collection_name, embedding_function=embedding_function
    )

    # We use a simple input loop.
    while True:
        # Get the user's query
        query = input("Query: ")
        if len(query) == 0:
            print("Please enter a question. Ctrl+C to Quit.\n")
            continue
        print("\nThinking...\n")

        # Query the collection to get the 5 most relevant results
        results = collection.query(
            query_texts=[query], n_results=5, include=["documents", "metadatas"]
        )
        if not results.get("documents") or not results["documents"][0]:
            print("No results retrieved.")
            continue

        sources = "\n".join(
            [
                f"{result['filename']}: line {result['line_number']}"
                for result in results["metadatas"][0]  # type: ignore
            ]
        )

        # Get the response from Gemini
        response = get_gemini_response(query, results["documents"][0])  # type: ignore

        # Output, with sources
        print(response)
        print("\n")
        print(f"Source documents:\n{sources}")
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load documents from a directory into a Chroma collection"
    )

    parser.add_argument(
        "--persist_directory",
        type=str,
        default="chroma_storage",
        help="The directory where you want to store the Chroma collection",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="documents_collection",
        help="The name of the Chroma collection",
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        collection_name=args.collection_name,
        persist_directory=args.persist_directory,
    )
