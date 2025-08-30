from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List
import chromadb
from chromadb.config import Settings
from config import Config

class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents"):
        self.collection_name = collection_name
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            google_api_key=Config.GOOGLE_API_KEY
        )
        
        # Initialize ChromaDB Cloud client
        self.chroma_client = chromadb.CloudClient(
            api_key=Config.CHROMA_API_KEY,
            tenant=Config.CHROMA_TENANT,
            database=Config.CHROMA_DATABASE
        )
        
        # Initialize Chroma vector store
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize the Chroma vector store"""
        # Verify all required credentials are present
        if not Config.CHROMA_API_KEY:
            raise ValueError("CHROMA_API_KEY not found in environment variables")
        if not Config.CHROMA_TENANT:
            raise ValueError("CHROMA_TENANT not found in environment variables")
        if not Config.CHROMA_DATABASE:
            raise ValueError("CHROMA_DATABASE not found in environment variables")
        
        try:
            # Test the connection first
            print("Testing Chroma Cloud connection...")
            collections = self.chroma_client.list_collections()
            print(f"Successfully connected to Chroma Cloud. Found {len(collections)} existing collections.")
            
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            print(f"Connected to Chroma Cloud collection: {self.collection_name}")
        except Exception as e:
            print(f"Error connecting to Chroma Cloud: {str(e)}")
            print("Please check your Chroma Cloud credentials in .env file")
            raise
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        try:
            self.vector_store.add_documents(documents)
            print(f"Successfully added {len(documents)} documents to vector store")
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = Config.TOP_K) -> List[Document]:
        """Perform similarity search"""
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error during similarity search: {str(e)}")
            return []
    
    def get_retriever(self, k: int = Config.TOP_K):
        """Get retriever object for use with chains"""
        return self.vector_store.as_retriever(search_kwargs={"k": k})
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {str(e)}")
    
    def test_connection(self):
        """Test the Chroma Cloud connection"""
        try:
            print("Testing Chroma Cloud connection...")
            print(f"API Key: {'*' * (len(Config.CHROMA_API_KEY) - 4) + Config.CHROMA_API_KEY[-4:]}")
            print(f"Tenant: {Config.CHROMA_TENANT}")
            print(f"Database: {Config.CHROMA_DATABASE}")
            
            collections = self.chroma_client.list_collections()
            print(f"Connection successful! Found {len(collections)} collections:")
            for collection in collections:
                print(f"  - {collection.name}")
            return True
        except Exception as e:
            print(f"Connection failed: {str(e)}")
            return False
    
    def get_collection_info(self):
        """Get information about the collection"""
        try:
            collection = self.chroma_client.get_collection(name=self.collection_name)
            return {
                "name": collection.name,
                "count": collection.count()
            }
        except Exception as e:
            print(f"Error getting collection info: {str(e)}")
            return None