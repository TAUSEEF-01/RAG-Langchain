from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from vector_store import VectorStore
from config import Config

class QueryEngine:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=0.1
        )
        
        # Custom prompt template
        self.prompt_template = PromptTemplate(
            template="""Use the following pieces of context to answer the question at the end. 
            If you don't know the answer based on the context provided, just say that you don't know, 
            don't try to make up an answer.

            Context:
            {context}

            Question: {question}

            Answer: """,
            input_variables=["context", "question"]
        )
        
        # Initialize retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.get_retriever(),
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True
        )
    
    def query(self, question: str):
        """Process a query and return answer with sources"""
        try:
            result = self.qa_chain.invoke({"query": question})
            
            response = {
                "answer": result["result"],
                "sources": []
            }
            
            # Extract source information
            for doc in result["source_documents"]:
                source_info = {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                }
                response["sources"].append(source_info)
            
            return response
        
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return {
                "answer": "Sorry, I encountered an error while processing your query.",
                "sources": []
            }
    
    def simple_query(self, question: str) -> str:
        """Simple query that returns just the answer"""
        result = self.query(question)
        return result["answer"]