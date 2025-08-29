import streamlit as st
import time
import asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


from dotenv import load_dotenv

load_dotenv()

# Ensure an asyncio event loop exists in the Streamlit script thread (fixes: RuntimeError no current event loop)
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

st.title("RAG Application built on Gemini Model")

# Cache heavy objects so they are created only once per session
if "retriever" not in st.session_state:
    loader = PyPDFLoader("yolov9_paper.pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)
    st.session_state["docs_count"] = len(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    st.session_state["retriever"] = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 10}
    )
    st.session_state["llm"] = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None
    )

retriever = st.session_state["retriever"]
llm = st.session_state["llm"]
st.caption(f"Loaded {st.session_state.get('docs_count', '?')} document chunks.")


query = st.chat_input("Say something: ")
prompt = query

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})
    # print(response["answer"])

    st.write(response["answer"])
