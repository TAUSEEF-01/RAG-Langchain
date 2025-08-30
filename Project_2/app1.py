import streamlit as st
import time
import os
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

st.title("RAG Application built on Gemini Model (Persistent Chroma)")

# Cache heavy objects so they are created only once per session
if "retriever" not in st.session_state:
    persist_dir = "chroma_store"  # existing directory with chroma.sqlite3
    rebuild = st.sidebar.toggle(
        "Rebuild Vector Store from PDF",
        value=False,
        help="Force re-embedding Automata.pdf and overwrite the existing Chroma store.",
    )

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    store_exists = os.path.exists(os.path.join(persist_dir, "chroma.sqlite3"))

    if store_exists and not rebuild:
        vectorstore = Chroma(
            persist_directory=persist_dir, embedding_function=embeddings
        )
        st.session_state["docs_count"] = vectorstore._collection.count()  # type: ignore (private attr)
        st.sidebar.success("Loaded existing Chroma store")
    else:
        with st.spinner("Building / Rebuilding vector store from PDF ..."):
            loader = PyPDFLoader("Automata.pdf")
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=150
            )
            docs = text_splitter.split_documents(data)
            st.session_state["docs_count"] = len(docs)
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=persist_dir,
            )
            vectorstore.persist()
        st.sidebar.success("Vector store (re)built")

    st.session_state["vectorstore"] = vectorstore
    st.session_state["retriever"] = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    )
    st.session_state["llm"] = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", temperature=0, max_tokens=None, timeout=None
    )

retriever = st.session_state["retriever"]
llm = st.session_state["llm"]
st.caption(f"Indexed chunks: {st.session_state.get('docs_count', '?')}")


query = st.chat_input("Ask a question about the embedded PDF:")
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

    with st.spinner("Retrieving & generating answer..."):
        response = rag_chain.invoke({"input": query})

    st.markdown("### Answer")
    st.write(response["answer"])

    # Show sources / context docs
    context_docs = response.get("context", [])
    if context_docs:
        with st.expander("Show retrieved context (sources)"):
            for i, d in enumerate(context_docs, 1):
                meta = d.metadata or {}
                source = meta.get("source") or meta.get("file_path") or "(unknown)"
                st.markdown(
                    f"**{i}. Source:** {source}\n\n````\n{d.page_content[:1200]}\n````"
                )
    else:
        st.caption("No context documents returned.")

    st.divider()
    st.caption(
        "Tip: Use the sidebar toggle to rebuild the vector store if underlying documents change."
    )
