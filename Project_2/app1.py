import streamlit as st
import os
import asyncio
import requests
import pathlib
import hashlib
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

# ----------------------- Sidebar: PDF Source Selection -----------------------
st.sidebar.header("PDF Source")


def save_uploaded_pdf(uploaded_file) -> pathlib.Path:
    uploads_dir = pathlib.Path("uploaded_pdfs")
    uploads_dir.mkdir(exist_ok=True)
    safe_name = pathlib.Path(uploaded_file.name).name
    target = uploads_dir / safe_name
    with open(target, "wb") as f:
        f.write(uploaded_file.read())
    latest = uploads_dir / "uploaded_latest.pdf"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        try:
            os.symlink(target, latest)
        except OSError:
            import shutil

            shutil.copy2(target, latest)
    except Exception:
        pass
    return target


uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF", type=["pdf"], help="Embed a local PDF."
)
pdf_url = st.sidebar.text_input(
    "Or PDF URL", placeholder="https://example.com/file.pdf"
)
download_button = st.sidebar.button("Download URL PDF", disabled=not pdf_url)

chosen_pdf_path: pathlib.Path | None = None
if uploaded_pdf is not None:
    try:
        chosen_pdf_path = save_uploaded_pdf(uploaded_pdf)
        st.sidebar.success(f"Uploaded: {uploaded_pdf.name}")
    except Exception as e:
        st.sidebar.error(f"Upload failed: {e}")
elif download_button and pdf_url:
    try:
        resp = requests.get(pdf_url, timeout=60)
        resp.raise_for_status()
        dl_dir = pathlib.Path("downloaded_pdfs")
        dl_dir.mkdir(exist_ok=True)
        hash_id = hashlib.sha1(pdf_url.encode()).hexdigest()[:12]
        filename = f"url_{hash_id}.pdf"
        target = dl_dir / filename
        target.write_bytes(resp.content)
        latest = dl_dir / "uploaded_latest.pdf"
        try:
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            try:
                os.symlink(target, latest)
            except OSError:
                import shutil

                shutil.copy2(target, latest)
        except Exception:
            pass
        chosen_pdf_path = target
        st.sidebar.success(f"Downloaded PDF -> {filename}")
    except Exception as e:
        st.sidebar.error(f"Download failed: {e}")

if chosen_pdf_path is None:
    # fallback to most recent uploaded or default Automata.pdf
    fallback = pathlib.Path("uploaded_pdfs/uploaded_latest.pdf")
    if fallback.exists():
        chosen_pdf_path = fallback
    else:
        chosen_pdf_path = pathlib.Path("Automata.pdf")

st.sidebar.caption(f"Active PDF: {chosen_pdf_path}")
st.session_state["active_pdf_path"] = str(chosen_pdf_path)
os.environ["PDF_PATH"] = str(chosen_pdf_path)  # for notebook / external processes

rebuild = st.sidebar.toggle(
    "Rebuild Vector Store from PDF",
    value=False,
    help="Force re-embedding current PDF and overwrite/create its Chroma store.",
)

pdf_hash = hashlib.sha1(str(chosen_pdf_path).encode()).hexdigest()[:10]
persist_dir = f"chroma_store_{pdf_hash}"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
store_exists = os.path.exists(os.path.join(persist_dir, "chroma.sqlite3"))

need_build = (
    rebuild
    or ("retriever" not in st.session_state)
    or st.session_state.get("_retriever_pdf_hash") != pdf_hash
    or not store_exists
)

if need_build:
    with st.spinner("Loading & embedding PDF ..."):
        loader = PyPDFLoader(str(chosen_pdf_path))
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
        # Safely persist (handle versions lacking persist or write issues)
        try:
            if hasattr(vectorstore, "persist"):
                vectorstore.persist()
        except Exception as e:
            st.sidebar.warning(f"Persist skipped: {e}")
    st.session_state["vectorstore"] = vectorstore
    st.session_state["retriever"] = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    )
    st.session_state["llm"] = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", temperature=0, max_tokens=None, timeout=None
    )
    st.session_state["_retriever_pdf_hash"] = pdf_hash
    st.sidebar.success("Vector store ready")
else:
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    st.session_state["vectorstore"] = vectorstore
    st.session_state["retriever"] = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    )
    st.sidebar.info("Using cached vector store")

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
