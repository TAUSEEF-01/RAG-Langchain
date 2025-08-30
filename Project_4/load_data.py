import os
import argparse
import time

from tqdm import tqdm

import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# Auto-load .env if python-dotenv is available
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    pass


def _read_text_lines(
    path: str, encodings: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
):
    """
    Try multiple encodings to read a text file. Final fallback ignores undecodable bytes.
    """
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.readlines()
        except UnicodeDecodeError as e:
            last_err = e
            continue
    # Fallback: ignore problematic bytes
    with open(path, "r", encoding=encodings[0], errors="ignore") as f:
        return f.readlines()


def main(
    documents_directory: str = "documents",
    collection_name: str = "documents_collection",
    persist_directory: str = ".",
    batch_size: int = 100,
) -> None:
    # Read all files in the data directory
    documents = []
    metadatas = []
    files = os.listdir(documents_directory)
    for filename in files:
        full_path = os.path.join(documents_directory, filename)
        if not os.path.isfile(full_path):
            continue
        try:
            lines = _read_text_lines(full_path)
        except OSError as e:
            print(f"Skipping {filename}: {e}")
            continue
        for line_number, line in enumerate(tqdm(lines, desc=f"Reading {filename}"), 1):
            line = line.strip()
            # Skip empty lines
            if len(line) == 0:
                continue
            documents.append(line)
            metadatas.append({"filename": filename, "line_number": line_number})

    # Instantiate a persistent chroma client in the persist_directory.
    # Learn more at docs.trychroma.com
    # client = chromadb.PersistentClient(path=persist_directory)

    # Load Chroma Cloud credentials from environment (.env already loaded above)
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

    google_api_key = None
    if "GOOGLE_API_KEY" not in os.environ:
        gapikey = input("Please enter your Google API Key: ")
        genai.configure(api_key=gapikey)
        google_api_key = gapikey
    else:
        google_api_key = os.environ["GOOGLE_API_KEY"]
        genai.configure(api_key=google_api_key)

    # create embedding function (correct class name + model override)
    embed_model = os.getenv("EMBED_MODEL", "models/embedding-001")
    try:
        embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=google_api_key,
            model_name=embed_model,
        )
    except AttributeError as e:
        raise RuntimeError(
            "chromadb version missing GoogleGenerativeAiEmbeddingFunction. Upgrade chromadb."
        ) from e
    print(f"Using embedding model: {embed_model}")

    # If the collection already exists, we just return it. This allows us to add more
    # data to an existing collection.
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_function
    )

    # Create ids from the current count
    count = collection.count()
    print(f"Collection already contains {count} documents")
    ids = [str(i) for i in range(count, count + len(documents))]

    if batch_size <= 0:
        batch_size = 100

    total = len(documents)
    print(f"Preparing to add {total} documents in batches of {batch_size}")

    start_all = time.time()
    for start in tqdm(range(0, total, batch_size), desc="Adding documents"):
        end = start + batch_size
        batch_ids = ids[start:end]
        batch_docs = documents[start:end]
        batch_meta = metadatas[start:end]  # type: ignore
        t0 = time.time()
        try:
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
            )
        except Exception as e:
            print(f"\nError adding batch {start//batch_size + 1}: {e}")
            print("Retrying once after 2s...")
            time.sleep(2)
            try:
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_meta,
                )
            except Exception as e2:
                print(
                    f"Failed batch permanently: {e2}. Skipping these {len(batch_ids)} docs."
                )
                continue
        dt = time.time() - t0
        print(
            f"Batch {start//batch_size + 1} ({len(batch_ids)} docs) added in {dt:.2f}s"
        )
    total_dt = time.time() - start_all
    new_count = collection.count()
    print(
        f"Added {new_count - count} documents in {total_dt:.2f}s (final collection size {new_count})"
    )


if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(
        description="Load documents from a directory into a Chroma collection"
    )

    # Add arguments
    parser.add_argument(
        "--data_directory",
        type=str,
        default="documents",
        help="The directory where your text files are stored",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="documents_collection",
        help="The name of the Chroma collection",
    )
    parser.add_argument(
        "--persist_directory",
        type=str,
        default="chroma_storage",
        help="The directory where you want to store the Chroma collection",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of documents per Chroma add() call",
    )

    # Parse arguments
    args = parser.parse_args()
    main(
        documents_directory=args.data_directory,
        collection_name=args.collection_name,
        persist_directory=args.persist_directory,
        batch_size=args.batch_size,
    )
