import os
from document_processor import DocumentProcessor
from vector_store import VectorStore
from query_engine import QueryEngine

class RAGSystem:
    def __init__(self, collection_name: str = "pdf_documents"):
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore(collection_name)
        self.query_engine = None
    
    def upload_pdfs(self, pdf_paths):
        """Upload and process PDF files"""
        print("Starting PDF processing...")
        
        # Load PDFs
        documents = self.doc_processor.load_multiple_pdfs(pdf_paths)
        
        if not documents:
            print("No documents were loaded successfully.")
            return False
        
        # Split into chunks
        chunks = self.doc_processor.split_documents(documents)
        
        # Add to vector store
        self.vector_store.add_documents(chunks)
        
        # Initialize query engine
        self.query_engine = QueryEngine(self.vector_store)
        
        print("PDF processing completed successfully!")
        return True
    
    def query(self, question: str):
        """Query the RAG system"""
        if not self.query_engine:
            return "Please upload PDFs first before querying."
        
        return self.query_engine.query(question)
    
    def get_collection_info(self):
        """Get information about the current collection"""
        return self.vector_store.get_collection_info()

def main():
    # Initialize RAG system
    print("Initializing RAG System...")
    try:
        rag = RAGSystem("my_pdf_collection")
        
        # Test Chroma Cloud connection
        if not rag.vector_store.test_connection():
            print("Failed to connect to Chroma Cloud. Please check your credentials.")
            return
        
    except Exception as e:
        print(f"Failed to initialize RAG system: {str(e)}")
        return
    
    while True:
        print("\n=== RAG System Menu ===")
        print("1. Upload PDFs")
        print("2. Query documents")
        print("3. Collection info")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            pdf_folder = input("Enter PDF folder path (or single PDF path): ").strip()
            
            if os.path.isfile(pdf_folder) and pdf_folder.endswith('.pdf'):
                # Single PDF file
                pdf_paths = [pdf_folder]
            elif os.path.isdir(pdf_folder):
                # Directory with PDFs
                pdf_paths = [
                    os.path.join(pdf_folder, f) 
                    for f in os.listdir(pdf_folder) 
                    if f.lower().endswith('.pdf')
                ]
            else:
                print("Invalid path provided.")
                continue
            
            if not pdf_paths:
                print("No PDF files found.")
                continue
            
            print(f"Found {len(pdf_paths)} PDF file(s)")
            rag.upload_pdfs(pdf_paths)
        
        elif choice == "2":
            if not rag.query_engine:
                print("Please upload PDFs first.")
                continue
            
            while True:
                question = input("\nEnter your question (or 'back' to return): ").strip()
                
                if question.lower() == 'back':
                    break
                
                if not question:
                    print("Please enter a valid question.")
                    continue
                
                print("\nProcessing your query...")
                result = rag.query(question)
                
                print(f"\nAnswer: {result['answer']}")
                
                if result['sources']:
                    print(f"\nSources ({len(result['sources'])}):")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"{i}. From {source['metadata'].get('source', 'Unknown')}")
                        print(f"   {source['content']}")
        
        elif choice == "3":
            info = rag.get_collection_info()
            if info:
                print(f"\nCollection: {info['name']}")
                print(f"Document count: {info['count']}")
            else:
                print("No collection information available.")
        
        elif choice == "4":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()