# test_chroma.py
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("CHROMA_API_KEY")
tenant = os.getenv("CHROMA_TENANT")
database = os.getenv("CHROMA_DATABASE")

print(f"API Key: {api_key[:10]}..." if api_key else "Not found")
print(f"Tenant: {tenant}")
print(f"Database: {database}")

try:
    client = chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database
    )
    
    collections = client.list_collections()
    print(f"Success! Found {len(collections)} collections")
    
except Exception as e:
    print(f"Error: {e}")