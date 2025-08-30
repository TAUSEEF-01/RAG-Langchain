# """Utility script to verify Chroma Cloud connectivity.

# Usage (PowerShell):
#   $env:CHROMA_API_KEY='YOUR_API_KEY'
#   python check_chroma_cloud.py

# Optional environment overrides:
#   CHROMA_TENANT   (default: 204facc0-6c88-4b9a-ad1b-a483f3ff4181)
#   CHROMA_DATABASE (default: Rag-java-fest-2025)
#   CHROMA_TEST_COLLECTION (default: connectivity_test)

# The script will:
#   1. Connect to Chroma Cloud.
#   2. Ensure a test collection exists.
#   3. Add a sample document only if the collection is empty.
#   4. Run a similarity search to confirm end‑to‑end insert + query.
#   5. Print a concise PASS / FAIL summary and exit code (0 on success, 1 on failure).
# """

# from __future__ import annotations

# import os
# import sys
# import time
# from typing import List

# import chromadb
# from chromadb.errors import ChromaError

# try:
#     from langchain_community.vectorstores import Chroma
#     from langchain_google_genai import GoogleGenerativeAIEmbeddings
#     from langchain.schema import Document
# except ImportError as e:
#     print(
#         "Missing dependencies. Ensure 'chromadb', 'langchain-community', and 'langchain-google-genai' are installed."
#     )
#     raise

# # API_KEY = os.getenv("CHROMA_API_KEY", "").strip()
# API_KEY = "ck-2ciTgRnju3zMvC1PnoGnXv6p9cfrxAR7JU5NdXPTEE5k"
# TENANT = os.getenv(
#     "CHROMA_TENANT", "204facc0-6c88-4b9a-ad1b-a483f3ff4181"
# )  # Default tenant
# DATABASE = os.getenv("CHROMA_DATABASE", "Rag-java-fest-2025")
# COLLECTION_NAME = os.getenv("CHROMA_TEST_COLLECTION", "index-ai-report-2025")
# EMBED_MODEL = os.getenv("CHROMA_TEST_EMBED_MODEL", "models/embedding-001")

# GOOGLE_API_KEY = "AIzaSyDiTrCDt9vZlZuZbN2E5U89iuwfmYufzwk"
# # (
# #     os.getenv("GOOGLE_API_KEY")
# #     or os.getenv("GEMINI_API_KEY")  # common alternate name
# #     or os.getenv("GOOGLE_GENAI_API_KEY")
# # )

# if not API_KEY:
#     print(
#         "ERROR: CHROMA_API_KEY not set. Set it (e.g. PowerShell: $env:CHROMA_API_KEY='ck-...') and re-run."
#     )
#     sys.exit(1)

# if not GOOGLE_API_KEY:
#     print(
#         "ERROR: GOOGLE_API_KEY (or GEMINI_API_KEY) not set. Set your Gemini API key so GoogleGenerativeAIEmbeddings can authenticate.\n"
#         "PowerShell example: $env:GOOGLE_API_KEY='your_key'"
#     )
#     sys.exit(1)

# start = time.time()
# print("Connecting to Chroma Cloud ...")
# try:
#     client = chromadb.CloudClient(api_key=API_KEY, tenant=TENANT, database=DATABASE)
# except Exception as e:  # broad: connectivity/auth
#     print(f"FAIL: Could not create CloudClient: {e}")
#     sys.exit(1)

# print(f"Connected. Tenant={TENANT} Database={DATABASE}")

# try:
#     import chromadb as _cdb
#     import langchain_community as _lc
#     import langchain_core as _lcc

#     print(
#         f"Versions: chromadb={_cdb.__version__} langchain-community={_lc.__version__} langchain-core={_lcc.__version__}"
#     )
# except Exception:
#     pass

# # Embedding function (explicitly pass API key to avoid falling back to ADC)
# try:
#     emb_fn = GoogleGenerativeAIEmbeddings(
#         model=EMBED_MODEL, google_api_key=GOOGLE_API_KEY
#     )
# except Exception as e:
#     print(f"FAIL: Could not initialize GoogleGenerativeAIEmbeddings: {e}")
#     sys.exit(1)

# # Instantiate (or create) vectorstore
# try:
#     # Create a LangChain Chroma wrapper pointing to the named collection.
#     # If the collection does not exist, Chroma.from_documents() below will create it.
#     # We first attempt lightweight get; if it fails we will create with from_documents.
#     create_new = False
#     try:
#         client.get_collection(COLLECTION_NAME)
#     except Exception as e:
#         create_new = True

#     if create_new:
#         print(
#             f"Collection '{COLLECTION_NAME}' not found. Creating and inserting sample document ..."
#         )
#         seed_texts = ["This is a connectivity test document for Chroma Cloud."]
#         try:
#             vs = Chroma.from_texts(
#                 texts=seed_texts,
#                 embedding=emb_fn,
#                 collection_name=COLLECTION_NAME,
#                 client=client,
#             )
#         except Exception as inner_e:
#             import traceback

#             print("DEBUG: Failed creating collection via from_texts")
#             traceback.print_exc()
#             raise
#     # List existing collections for diagnostics
#     try:
#         existing = [c.name for c in client.list_collections()]
#         print(f"Existing collections: {existing}")
#     except Exception as _e_list:
#         print(f"(WARN) Could not list collections: {_e_list}")
#     else:
#         print(f"Using existing collection '{COLLECTION_NAME}'.")
#         vs = Chroma(
#             collection_name=COLLECTION_NAME, embedding_function=emb_fn, client=client
#         )
#         # Add a sample doc only if empty
#         try:
#             count = client.get_collection(COLLECTION_NAME).count()
#         except Exception:
#             count = None
#         if not count:
#             print("Collection empty. Adding sample document ...")
#             vs.add_texts(["This is a connectivity test document for Chroma Cloud."])
# except ChromaError as ce:
#     import traceback

#     if os.getenv("CHROMA_DEBUG"):
#         traceback.print_exc()
#     print(f"FAIL: Chroma operation error: {ce}")
#     sys.exit(1)
# except Exception as e:
#     import traceback

#     if os.getenv("CHROMA_DEBUG"):
#         traceback.print_exc()
#     hint = ""
#     if "_type" in str(e):
#         hint = (
#             " Hint: '_type' often indicates a version mismatch (langchain-community/chromadb) or stale Document schema. "
#             "Run: pip install -U langchain-community langchain-core chromadb && clear old cached artifacts."
#         )
#     print(f"FAIL: Unexpected error preparing vector store: {e}.{hint}")
#     sys.exit(1)

# # Run a similarity search
# try:
#     query = "connectivity test"
#     results = vs.similarity_search(query, k=1)
#     if not results:
#         print("FAIL: Similarity search returned no results.")
#         sys.exit(1)
#     top = results[0]
#     print("Top result snippet:")
#     print(top.page_content[:120].replace("\n", " "), "...")
# except Exception as e:
#     print(f"FAIL: Similarity search failed: {e}")
#     sys.exit(1)

# duration = time.time() - start
# print(
#     f"PASS: End-to-end Chroma Cloud connectivity & query succeeded in {duration:.2f}s"
# )
# sys.exit(0)



import chromadb
import sys

def test_chroma_connection():
    try:
        print("Connecting to ChromaDB Cloud...")
        
        # Your connection details
        client = chromadb.CloudClient(
            api_key='ck-9NdWnh1f3QmVRMbKre5wigeL1tkP6spu8JovvDfxaAn7',
            tenant='204facc0-6c88-4b9a-ad1b-a483f3ff4181',
            database='Rag-java-fest-2025'
        )
        
        print("✓ Client created successfully")
        
        # Test basic connection
        try:
            print("Testing connection...")
            # Try to list collections first
            collections = client.list_collections()
            print(f"✓ Connection successful! Found {len(collections)} existing collections")
            
            # Print existing collections
            if collections:
                print("Existing collections:")
                for col in collections:
                    print(f"  - {col.name}")
        
        except Exception as e:
            print(f"✗ Connection test failed: {e}")
            return False
        
        # Try to get or create collection with explicit configuration
        collection_name = "my_collection"
        
        try:
            print(f"\nTrying to get existing collection '{collection_name}'...")
            collection = client.get_collection(name=collection_name)
            print(f"✓ Retrieved existing collection: {collection_name}")
            
        except Exception as get_error:
            print(f"Collection doesn't exist or couldn't be retrieved: {get_error}")
            
            try:
                print(f"Creating new collection '{collection_name}'...")
                collection = client.create_collection(
                    name=collection_name,
                    # You might need to specify metadata or configuration
                    metadata={"description": "Test collection for RAG project"}
                )
                print(f"✓ Created new collection: {collection_name}")
                
            except Exception as create_error:
                print(f"✗ Failed to create collection: {create_error}")
                
                # Try with get_or_create as a last resort with explicit error handling
                try:
                    print("Trying get_or_create_collection as fallback...")
                    collection = client.get_or_create_collection(
                        name=collection_name,
                        metadata={"description": "Test collection for RAG project"}
                    )
                    print("✓ get_or_create_collection succeeded")
                except Exception as final_error:
                    print(f"✗ All collection methods failed: {final_error}")
                    return False
        
        # Test adding documents
        try:
            print("\nTesting document operations...")
            collection.upsert(
                documents=[
                    "This is a document about pineapple",
                    "This is a document about oranges"
                ],
                ids=["id1", "id2"]
            )
            print("✓ Documents upserted successfully")
            
            # Test querying
            results = collection.query(
                query_texts=["This is a query document about florida"],
                n_results=2
            )
            print("✓ Query successful")
            print(f"Results: {results}")
            
        except Exception as doc_error:
            print(f"✗ Document operation failed: {doc_error}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Fatal error: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

def check_chromadb_version():
    try:
        print(f"ChromaDB version: {chromadb.__version__}")
    except AttributeError:
        print("ChromaDB version info not available")

if __name__ == "__main__":
    print("ChromaDB Cloud Connection Test")
    print("=" * 40)
    
    check_chromadb_version()
    print()
    
    success = test_chroma_connection()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Tests failed. Check the error messages above.")
        print("\nTroubleshooting suggestions:")
        print("1. Update ChromaDB: pip install --upgrade chromadb")
        print("2. Check your API credentials")
        print("3. Verify your tenant and database names")
        print("4. Try creating a collection with a different name")
        print("5. Contact ChromaDB support if the issue persists")