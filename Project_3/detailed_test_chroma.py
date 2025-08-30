import chromadb
from dotenv import load_dotenv
import os
import sys

# Load environment variables
load_dotenv()

# Get credentials
api_key = os.getenv("CHROMA_API_KEY")
tenant = os.getenv("CHROMA_TENANT") 
database = os.getenv("CHROMA_DATABASE")

print("=== Chroma Cloud Connection Test ===")
print(f"ChromaDB version: {chromadb.__version__}")
print(f"API Key: {'*' * 10}{api_key[-4:] if api_key else 'NOT FOUND'}")
print(f"Tenant: {tenant}")
print(f"Database: {database}")
print()

# Check if credentials exist
if not api_key:
    print("❌ ERROR: CHROMA_API_KEY not found in .env file")
    sys.exit(1)
    
if not tenant:
    print("❌ ERROR: CHROMA_TENANT not found in .env file")
    sys.exit(1)
    
if not database:
    print("❌ ERROR: CHROMA_DATABASE not found in .env file")
    sys.exit(1)

print("✅ All credentials found in .env file")
print()

# Test different connection approaches
print("=== Testing Connection Methods ===")

# Method 1: Standard connection
print("1. Testing standard CloudClient connection...")
try:
    client = chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database
    )
    print("   ✅ Client created successfully")
    
    # Test basic operations
    collections = client.list_collections()
    print(f"   ✅ Found {len(collections)} collections")
    
    if collections:
        for collection in collections:
            print(f"      - {collection.name}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    print(f"   Error type: {type(e).__name__}")

print()

# Method 2: Try without database parameter (some versions don't need it)
print("2. Testing connection without database parameter...")
try:
    client = chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant
    )
    print("   ✅ Client created successfully (no database param)")
    
    collections = client.list_collections()
    print(f"   ✅ Found {len(collections)} collections")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

print()

# Method 3: Check if it's a tenant vs database issue
print("3. Testing with different parameter combinations...")
try:
    # Some versions expect different parameter names
    client = chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database
    )
    
    # Try to create a test collection
    test_collection = client.get_or_create_collection(name="connection_test")
    print("   ✅ Successfully created/accessed test collection")
    print(f"   Collection has {test_collection.count()} documents")
    
except Exception as e:
    print(f"   ❌ Error creating collection: {e}")

print()
print("=== Debugging Information ===")
print("If you're still getting errors, please check:")
print("1. Your Chroma Cloud dashboard to verify the tenant ID")
print("2. That your API key has the correct permissions")
print("3. That the database name matches exactly (case-sensitive)")
print("4. Your internet connection")

# Additional debugging
try:
    print(f"\nEnvironment file path: {os.path.abspath('.env')}")
    print(f"Environment file exists: {os.path.exists('.env')}")
    
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            lines = f.readlines()
        print(f"Environment file has {len(lines)} lines")
        for i, line in enumerate(lines, 1):
            if 'CHROMA' in line:
                # Hide the actual values for security
                key = line.split('=')[0]
                print(f"   Line {i}: {key}=***")
                
except Exception as e:
    print(f"Error reading .env file: {e}")