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

st.title("RAG Application built on Gemini Model (Persistent Chroma, Multi-PDF)")

# ----------------------- Sidebar: PDF Source Selection -----------------------
st.sidebar.header("PDF Sources")


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


uploaded_pdfs = st.sidebar.file_uploader(
    "Upload PDFs (multiple)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Embed one or more local PDFs.",
)
pdf_urls_raw = st.sidebar.text_area(
    "Or PDF URLs (one per line)",
    placeholder="https://example.com/a.pdf\nhttps://example.com/b.pdf",
)
download_urls = st.sidebar.button(
    "Download URL PDFs", disabled=not pdf_urls_raw.strip()
)

chosen_pdf_paths: list[pathlib.Path] = []
# Handle uploads
if uploaded_pdfs:
    for uf in uploaded_pdfs:
        try:
            p = save_uploaded_pdf(uf)
            chosen_pdf_paths.append(p)
        except Exception as e:
            st.sidebar.error(f"Upload failed for {uf.name}: {e}")
    if uploaded_pdfs:
        st.sidebar.success(f"Uploaded {len(uploaded_pdfs)} PDF(s)")

# Handle URL downloads
if download_urls:
    dl_dir = pathlib.Path("downloaded_pdfs")
    dl_dir.mkdir(exist_ok=True)
    for line in pdf_urls_raw.splitlines():
        url = line.strip()
        if not url:
            continue
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            hash_id = hashlib.sha1(url.encode()).hexdigest()[:12]
            fname = f"url_{hash_id}.pdf"
            tgt = dl_dir / fname
            tgt.write_bytes(r.content)
            chosen_pdf_paths.append(tgt)
        except Exception as e:
            st.sidebar.error(f"Download failed for {url}: {e}")
    if download_urls and chosen_pdf_paths:
        st.sidebar.success("Downloaded URL PDF(s)")

# Fallback if no new selections
if not chosen_pdf_paths:
    # Try previously active list
    prev_list = st.session_state.get("active_pdf_paths")
    if prev_list:
        chosen_pdf_paths = [pathlib.Path(p) for p in prev_list]
    else:
        default_path = pathlib.Path("Automata.pdf")
        chosen_pdf_paths = [default_path]

# Deduplicate while preserving order
seen = set()
deduped = []
for p in chosen_pdf_paths:
    key = str(p.resolve())
    if key not in seen and p.exists():
        seen.add(key)
        deduped.append(p)
chosen_pdf_paths = deduped

st.sidebar.caption(
    "Active PDFs:\n" + "\n".join(f"- {p.name}" for p in chosen_pdf_paths)
)
st.session_state["active_pdf_paths"] = [str(p) for p in chosen_pdf_paths]
os.environ["PDF_PATHS"] = "|".join(str(p) for p in chosen_pdf_paths)

rebuild = st.sidebar.toggle(
    "Rebuild Vector Store from PDF",
    value=False,
    help="Force re-embedding current PDF and overwrite/create its Chroma store.",
)

multi_sig = ";".join(
    sorted(f"{p.name}:{p.stat().st_size}" for p in chosen_pdf_paths if p.exists())
)
pdf_hash = hashlib.sha1(multi_sig.encode()).hexdigest()[:10]
persist_dir = f"chroma_store_multi_{pdf_hash}"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
store_exists = os.path.exists(os.path.join(persist_dir, "chroma.sqlite3"))

need_build = (
    rebuild
    or ("retriever" not in st.session_state)
    or st.session_state.get("_retriever_pdf_hash") != pdf_hash
    or not store_exists
)

if need_build:
    with st.spinner("Loading & embedding PDF(s) ..."):
        all_docs = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )
        for p in chosen_pdf_paths:
            try:
                loader = PyPDFLoader(str(p))
                data = loader.load()
                docs = text_splitter.split_documents(data)
                # Tag metadata with source filename for traceability
                for d in docs:
                    d.metadata = d.metadata or {}
                    d.metadata["source_file"] = p.name
                all_docs.extend(docs)
            except Exception as e:
                st.sidebar.error(f"Failed to process {p.name}: {e}")
        st.session_state["docs_count"] = len(all_docs)
        vectorstore = Chroma.from_documents(
            documents=all_docs,
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
st.caption(
    f"Indexed chunks across {len(chosen_pdf_paths)} PDF(s): {st.session_state.get('docs_count', '?')}"
)


query = st.chat_input("Ask a question about the embedded PDF(s):")
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
                source = (
                    meta.get("source_file")
                    or meta.get("source")
                    or meta.get("file_path")
                    or "(unknown)"
                )
                st.markdown(
                    f"**{i}. Source:** {source}\n\n````\n{d.page_content[:1200]}\n````"
                )
    else:
        st.caption("No context documents returned.")

    st.divider()
    st.caption(
        "Tip: Use the sidebar toggle to rebuild the vector store if underlying documents change."
    )
