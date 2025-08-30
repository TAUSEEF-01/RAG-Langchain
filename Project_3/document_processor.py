from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import os
from config import Config

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load and process a single PDF file"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata["source"] = os.path.basename(pdf_path)
            doc.metadata["file_type"] = "pdf"
        
        return documents
    
    def load_multiple_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        """Load and process multiple PDF files"""
        all_documents = []
        
        for pdf_path in pdf_paths:
            try:
                docs = self.load_pdf(pdf_path)
                all_documents.extend(docs)
                print(f"Loaded {len(docs)} pages from {pdf_path}")
            except Exception as e:
                print(f"Error loading {pdf_path}: {str(e)}")
        
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
        
        print(f"Split documents into {len(chunks)} chunks")
        return chunks