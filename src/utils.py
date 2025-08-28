import time
import sys
import os

import cohere
from typing import Optional, Callable

# Adding project path to sys to solve importing errors
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from langchain_core.messages import AIMessage
from langchain_core.agents import AgentStep

# local
from src.data_loading import DataLoader
from src.embeddings.embeddings_factory import EmbeddingsFactory
from src.llms.llm_factory import LLMFactory
from src.chunking import TextSplitter
from src.indexing import Index
from src.rag_agent import ReActRagAgent


def response_generator(text: str):
    """
    Generate characters from the input text one by one with a small delay, simulating a typing effect.

    This function yields each character in the input text with a delay of 0.002 seconds between each character.
    making it useful for creating a simulated typing response in an interactive environment.

    Args:
        - text (str): The input text to be processed and yielded character by character.

    Yields:
        - str: The next character from the input text, one at a time.

    Example:
        >>> for char in response_generator("Hello World"):
        >>>     print(char, end="")
        H e l l o   W o r l d
    """

    for char in text:
        yield char
        time.sleep(0.00001)


def load_pdf(pdf_file_path: str):
    """
    Load and extract documents from a PDF file.

    This function initializes a `DataLoader` instance with the given PDF file path,
    loads the documents from the PDF, and returns the extracted documents.

    Args:
        - pdf_file_path (str): The file path to the PDF document to be loaded.

    Returns:
        - list: A list of `Document` objects extracted from the PDF file.
    """

    loader = DataLoader(file_path=pdf_file_path)
    docs = loader.get_docs()

    return docs


def create_embeddings(embedding_type_selection: str, **kwargs):
    """
    Create and return embeddings based on the selected embedding model.

    This function selects the appropriate embedding model (Cohere or HuggingFace)
    based on the `embedding_type_selection` argument. It asserts the necessary
    arguments are passed (such as the `cohere_api_key` for Cohere) and then
    returns the corresponding embeddings.

    Args:
        - embedding_type_selection (str): The embedding model selection.
                                        Should be either "cohere" or "huggingface".
        - **kwargs: Additional arguments required for embedding model initialization.
                  For Cohere, it requires `cohere_api_key`.

    Returns:
        - The embeddings object corresponding to the selected model.
    """

    provider = None
    if embedding_type_selection.startswith("cohere"):
        provider = "cohere"
        assert "cohere_api_key" in kwargs, "Please, pass the `cohere_api_key` argument"
    elif embedding_type_selection.startswith("models/text-embedding"):
        provider = "gemini"
        assert "google_api_key" in kwargs, "Please, pass the `google_api_key` argument"
    else:
        provider = "huggingface"

    embeddings = EmbeddingsFactory.create_embeddings(provider, **kwargs)

    return embeddings.get_embeddings()


def split_documents(docs):
    """
    Split documents into smaller chunks.

    This function uses a text splitter to divide the input documents into smaller chunks
    for easier processing. It applies the splitting logic defined in the `TextSplitter` class.

    Args:
        - docs (list): A list of `Document` objects to be split.

    Returns:
        - list: A list of smaller document chunks after splitting the original documents.
    """

    text_splitter = TextSplitter()
    chunks = text_splitter.split_documents(docs)
    return chunks


def build_vector_store(embeddings):
    """
    Build and return a vector store using the provided embeddings.

    This function initializes an `Index` object with the given embeddings and
    returns the vector store associated with it.

    Args:
        - embeddings: The embeddings function or model used for creating the vector store.

    Returns:
        - vector_store: The created vector store, which holds the embedded documents.
    """

    index = Index(embeddings)

    return index.vector_store


def create_rag_agent_exectutor(
    provider_api_key: str,
    vector_store,
    number_of_retrieved_documents: int,
    llm_provider: str = "cohere",
):
    """
    Creates a ReAct RAG agent executor using the provided Cohere API key, vector store,
    and the number of retrieved documents.

    This function initializes a Cohere LLM using the provided API key, sets up the ReAct RAG agent with
    the LLM and vector store, and creates an agent executor that can handle queries using the Retrieve-Ask-Generate
    approach.

    Args:
        - cohere_api_key (str): The API key for accessing the Cohere API.
        - vector_store (langchain.vectorstores.VectorStore): The vector store to be used for document retrieval.
        - number_of_retrieved_documents (int): The number of documents to retrieve during the "Retrieve" phase.

    Returns:
        - langchain.agents.AgentExecutor: The created RAG agent executor for processing queries.

    Example:
        >>> rag_executor = create_rag_agent_exectutor(
                cohere_api_key="your_cohere_api_key",
                vector_store=my_vector_store,
                number_of_retrieved_documents=5
            )
        >>> response = rag_executor.run("What is the impact of AI on healthcare?")
    """

    if llm_provider == "cohere":
        llm_wrapper = LLMFactory.create_llm("cohere", cohere_api_key=provider_api_key)
    elif llm_provider == "gemini":
        llm_wrapper = LLMFactory.create_llm("gemini", google_api_key=provider_api_key)
    else:
        raise ValueError("Unsupported LLM provider")

    llm = llm_wrapper.get_llm()
    react_rag_agent = ReActRagAgent(llm, vector_store, number_of_retrieved_documents)
    rag_agent_executor = react_rag_agent.create_react_agent_executor()

    return rag_agent_executor


def process_query(
    query: str,
    provider_api_key: str,
    agent_avatar: str,
    number_of_retrieved_documents: int = 5,
    llm_provider: str = "cohere",
    simple_mode: bool = False,
):
    """
    Processes a user's query using a ReAct RAG agent, retrieving relevant
    information from the AI Index Report 2025 and generating a response using Cohere's LLM.

    This function checks if the required inputs are provided, ensures the RAG agent executor is set up,
    and then either generates a response or provides detailed reasoning behind the response. The reasoning
    includes invoking relevant tools and displaying the data retrieved for the user's query.

    Args:
        - query (str): The user's input query that the agent will respond to.
        - cohere_api_key (str): The API key used to access the Cohere API.
        - agent_avatar (str): The avatar image or icon for the assistant.
        - number_of_retrieved_documents (int, optional): The number of documents to retrieve from the vector
                                                      store. Defaults to 5.

    Returns:
        - None: This function does not return anything, but updates the session state with the generated
              response and reasoning steps if enabled.

    Example:
        >>> process_query("What are the key findings in the 2025 AI Index Report?",
        >>>                cohere_api_key="your_cohere_api_key",
        >>>                agent_avatar="avatar_image_path")
    """

    if not provider_api_key:
        st.error("⚠️ Please enter your API key in the sidebar.")
        return

    if not st.session_state.vector_store:
        st.error("⚠️ Please upload the AI Index Report 2025 PDF first.")
        return

    # If simple retrieval mode: bypass agent and do direct retrieve + synthesize
    if simple_mode:
        vector_store = st.session_state.vector_store
        # Determine retrieval query (rewrite short follow-ups using last substantive query)
        retrieval_query = query
        messages_without_current = st.session_state.messages[
            :-1
        ]  # exclude current user query
        last_substantive_query = None

        # Find the most recent substantive query (6+ words)
        for m in reversed(messages_without_current):
            if m.__class__.__name__ == "HumanMessage":
                content = m.content.strip()
                if len(content.split()) >= 6:  # substantive query
                    last_substantive_query = content
                    break

        # If current query is short/vague and we have a substantive previous query
        vague_patterns = [
            "explain",
            "again",
            "more",
            "detail",
            "bullet",
            "summarize",
            "what about",
            "tell me",
        ]
        is_vague = len(query.split()) < 6 and any(
            pattern in query.lower() for pattern in vague_patterns
        )

        if is_vague and last_substantive_query:
            # Use the substantive query for retrieval to get same context
            retrieval_query = last_substantive_query
            st.info(
                f"🔍 Using previous query for retrieval: '{last_substantive_query}'"
            )

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": number_of_retrieved_documents,
                "fetch_k": number_of_retrieved_documents * 4,
            },
        )
        docs = retriever.get_relevant_documents(retrieval_query)
        context_blocks = []
        for d in docs:
            page = d.metadata.get("page", "?")
            snippet = d.page_content.strip().replace("\n", " ")[:1200]
            context_blocks.append(f'(p. {page}) "{snippet}"')
        context = "\n".join(context_blocks)

        # Conversation history (last 6 exchanges) for contextual continuity
        history_messages = []
        # Use last 6 prior messages (excluding current user query) for history
        for m in messages_without_current[-6:]:
            role = "User" if m.__class__.__name__ == "HumanMessage" else "Assistant"
            trimmed = m.content.strip().replace("\n", " ")[:400]
            history_messages.append(f"{role}: {trimmed}")
        history_block = "\n".join(history_messages)
        history_section = (
            "Previous conversation (use only if consistent with excerpts):\n"
            + history_block
            + "\n\n"
            if history_block
            else ""
        )

        # Build or reuse LLM
        if llm_provider == "cohere":
            llm_wrapper = LLMFactory.create_llm(
                "cohere", cohere_api_key=provider_api_key
            )
        else:
            llm_wrapper = LLMFactory.create_llm(
                "gemini", google_api_key=provider_api_key
            )
        llm = llm_wrapper.get_llm()
        synthesis_prompt = (
            "You are the 2025 AI Index Report Assistant.\n"
            + history_section
            + "Answer using ONLY the provided excerpts from the 2025 AI Index Report.\n"
            "If the current question asks for reformatting (bullet points, summary, etc.) of a previous topic:\n"
            "  - Check if that topic is actually present in the current excerpts\n"
            "  - If yes, reformat the information from the excerpts as requested\n"
            "  - If no, acknowledge that the topic was previously discussed but clarify it's not in the report excerpts\n"
            "If the question is vague (like 'explain again'), refer to the conversation history to understand the topic, then:\n"
            "  - If the topic is in the current excerpts, explain it based on the excerpts\n"
            "  - If not, acknowledge what was previously discussed but clarify the information is not found in this report\n"
            "Always cite supporting excerpts with page numbers like (p. X) when information IS found.\n"
            "Never hallucinate information not present in the excerpts.\n\n"
            f"Excerpts:\n{context}\n\nCurrent Question: {query}\nAnswer:"
        )
        with st.chat_message("assistant", avatar=agent_avatar):
            with st.spinner("Generating Response..."):
                try:
                    answer = llm.invoke(synthesis_prompt).content
                except Exception as e:
                    answer = f"Error generating answer: {e}"
            st.write_stream(response_generator(answer))
            st.session_state.messages.append(AIMessage(answer))
        return

    # Agent mode
    if st.session_state.rag_agent_executer is None:
        st.session_state.rag_agent_executer = create_rag_agent_exectutor(
            provider_api_key,
            st.session_state.vector_store,
            number_of_retrieved_documents,
            llm_provider=llm_provider,
        )

    rag_agent_executer = st.session_state.rag_agent_executer
    if not rag_agent_executer:
        return

    with st.chat_message("assistant", avatar=agent_avatar):
        if st.session_state.show_reasoning:
            with st.spinner("Thinking..."):
                tool_calls = []
                response = rag_agent_executer.stream({"input": query})
                for event in response:
                    for _, value in event.items():
                        for step in value:
                            if isinstance(step, AgentStep):
                                action = step.action
                                if action.tool_input not in tool_calls:
                                    with st.expander("Reasoning Step", expanded=True):
                                        st.markdown(
                                            f"`✅ {action.message_log[-1].content}`"
                                        )

                                        st.markdown("##### 🔧Tools:")
                                        st.markdown(
                                            f"⚙️ **Invoking Tool**: `{action.tool}` with `{action.tool_input}`"
                                        )
                                        st.write("---")

                                        st.markdown("##### 📄 Retrived Data:")
                                        st.markdown(f"{step.observation}")
                                        tool_calls.append(action.tool_input)

            st.write_stream(response_generator(event["output"]))
            st.session_state.messages.append(AIMessage(event["output"]))

        else:
            with st.spinner("Generating Response..."):
                response = rag_agent_executer.invoke({"input": query})

            output_text = response.get("output", "")

            # Fallback: if agent didn't retrieve (generic self-description) try manual retrieval
            if (
                "I am the 2025 AI Index Report Assistant" in output_text
                and len(response.get("intermediate_steps", [])) == 0
            ):
                retriever = rag_agent_executer.tools[0].retriever  # first tool
                docs = retriever.get_relevant_documents(query)
                context_blocks = []
                for d in docs:
                    page = d.metadata.get("page", "?")
                    snippet = d.page_content.strip().replace("\n", " ")[:800]
                    context_blocks.append(f'(p. {page}) "{snippet}"')
                context = "\n".join(context_blocks)
                synthesis_prompt = (
                    "You are answering based ONLY on the provided excerpts from the 2025 AI Index Report.\n"
                    "If the answer truly isn't present, say you cannot find it in the report.\n\n"
                    f"Excerpts:\n{context}\n\nQuestion: {query}\nAnswer:"
                )
                # Direct LLM call bypassing agent
                try:
                    synthesized = rag_agent_executer.agent.llm.invoke(
                        synthesis_prompt
                    ).content
                    output_text = synthesized
                except Exception:
                    pass

            st.write_stream(response_generator(output_text))
            st.session_state.messages.append(AIMessage(output_text))


def estimate_tokens(text: str):
    """
    Estimate the number of tokens in a given text string using the tokenizer from Cohere.

    The function uses the Cohere library to tokenize the provided text and return the number of tokens.
    This is useful for estimating how much input text will cost in terms of tokens when interacting
    with language models that have token limits.

    Args:
        - text (str): The input text string for which the number of tokens will be estimated.

    Returns:
        - int: The estimated number of tokens in the input text.

    Example:
        >>> estimate_tokens("Hello, how are you?")
        5
    """
    api_key = os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY")
    if api_key:
        try:
            co = cohere.Client()
            response = co.tokenize(text=text, model="command-a-03-2025")
            return len(response.tokens)
        except Exception:
            pass
    # Fallback heuristic: ~4 chars per token
    return max(1, len(text) // 4)


@st.cache_resource(show_spinner=False)
def process_chunks_with_rate_limit_cohere(
    _chunks,
    _vectorstore,
    batch_size=166,
    token_limit=90000,
    token_counter: Optional[Callable[[str], int]] = None,
):
    """
    Processes document chunks in batches, respecting Cohere's rate limit by waiting for the remainder
    of the minute between each batch. It adds documents to the provided vectorstore and displays
    progress in Streamlit.

    Args:
        - _chunks (list): List of document chunks (with `page_content`).
        - _vectorstore (object): Initialized vectorstore instance.
        - batch_size (int, optional): Number of chunks per batch.
        - token_limit (int, optional): Token limit per batch.

    Returns:
        - vectorstore (object): The vectorstore with added documents.
    """
    total_batches = (len(_chunks) + batch_size - 1) // batch_size
    progress_bar = st.progress(0)
    status_text = st.empty()
    countdown_placeholder = st.empty()

    # mark the start of the rate-limit window
    window_start = time.time()

    if token_counter is None:
        token_counter = estimate_tokens

    for i in range(0, len(_chunks), batch_size):
        current_batch = i // batch_size + 1
        batch = _chunks[i : i + batch_size]
        batch_tokens = sum(token_counter(doc.page_content) for doc in batch)

        status_text.markdown(
            f"🛠️ Processing Batch `[{current_batch}|{total_batches}]` with `{len(batch)}` docs (~{batch_tokens} tokens)"
        )

        if batch_tokens > token_limit:
            st.warning(f"⚠️ Batch too large (~{batch_tokens} tokens), splitting... ")
            for doc in batch:
                doc_tokens = token_counter(doc.page_content)
                st.markdown(f"🛠️ Processing single doc (~{doc_tokens} tokens)")
                _vectorstore.add_documents([doc])
                if doc_tokens > 10000:
                    time.sleep(5)
        else:
            _vectorstore.add_documents(batch)

        # update progress
        progress = min((i + batch_size) / len(_chunks), 1.0)
        progress_bar.progress(progress)

        # wait until 60 seconds have elapsed since window_start
        if i + batch_size < len(_chunks):
            elapsed = time.time() - window_start
            remaining = max(0, int(60 - elapsed))
            for sec in range(remaining, 0, -1):
                countdown_placeholder.info(
                    f"⌛ Waiting **{sec}**s to respect API rate limit"
                )
                time.sleep(1)
            countdown_placeholder.info("🔃 Resuming processing...")
            # reset window
            window_start = time.time()

    status_text.success("✅ Batches have been processed successfully!")
    return _vectorstore


@st.cache_resource(show_spinner=False)
def add_chunks_to_vector_store_hf_embeddings(_chunks, _vector_store):
    """
    Add a list of document chunks to the provided vector store using HuggingFace embeddings.

    This function processes the document chunks in batches and adds them to the given vector store.
    A progress bar and progress text are displayed in the Streamlit UI to show the status of the
    document addition process.

    Args:
        - _chunks (list): A list of document chunks to be added to the vector store.
        - _vector_store: The vector store object where the chunks will be added.

    Returns:
        - _vector_store: The updated vector store with the added document chunks.

    Raises:
        - Exception: If an error occurs while adding chunks to the vector store, an exception is raised and displayed in the Streamlit UI.
    """

    try:
        # * batch_size is set to 166 as this is the largest batch size supported by
        # * the HuggingFace embedding model
        batch_size = 166
        total_chunks = len(_chunks)
        progress_bar = st.progress(0)
        progress_text = st.empty()

        for start in range(0, total_chunks, batch_size):
            end = min(start + batch_size, total_chunks)
            batch = _chunks[start:end]

            # Add documents in batch
            _vector_store.add_documents(batch)
            progress = int((end) / total_chunks * 100)
            progress_bar.progress(progress)
            progress_text.markdown(f"Processing chunk: `[{end}/{total_chunks}]`")

        return _vector_store

    except Exception as e:
        st.error(e)


def reset_memory():
    """
    Reset the memory of the RAG agent executor stored in the Streamlit session state.

    This function checks if the `rag_agent_executer` exists in the Streamlit session state,
    and if so, clears its memory to reset the internal state. This can be useful for restarting
    a conversation or clearing previous interactions stored in memory.

    Args:
        - None

    Returns:
        - None

    Raises:
        - None: If the executor is not found in the session state, the function exits silently.
    """

    rag_agent_executer = st.session_state.rag_agent_executer
    if not rag_agent_executer:
        return

    rag_agent_executer.memory.clear()
