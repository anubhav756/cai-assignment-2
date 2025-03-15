import streamlit as st
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
import nltk
from rank_bm25 import BM25Okapi
import time

st.set_page_config(page_title="Financial RAG Chatbot", layout="wide")

def download_nltk_resource(resource_name):
    try:
        nltk.data.find(resource_name)
    except LookupError:
        # Download NLTK resource if not already.
        nltk.download(resource_name)

@st.cache_resource
def download_nltk_resources():
    """
    Downloads required NLTK resources (punkt and punkt_tab).
    Cached to prevent repeated downloads on app reloads.
    """
    download_nltk_resource('punkt')
    download_nltk_resource('punkt_tab')

download_nltk_resources()

@st.cache_resource
def load_resources(chunk_size):
    """
    Loads and caches all necessary resources:
    - Pre-computed document embeddings
    - FAISS index for efficient similarity search
    - Document chunks
    - Sentence Transformer model for generating query embeddings
    - Cross-Encoder model for reranking retrieved documents
    - Hugging Face pipeline for the SLM
    - BM25 model for lexical search
    """

    embeddings = np.load(f"embeddings_{chunk_size}.npy")
    index = faiss.read_index(f"embeddings_{chunk_size}.index")
    with open(f"chunks_{chunk_size}.json", "r") as f:
        documents = json.load(f)
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    slm_pipeline = pipeline('text-generation', model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', device_map="auto", torch_dtype='auto')

    # Initialize BM25 model
    tokenized_documents = [nltk.word_tokenize(doc['text'].lower()) for doc in documents]
    bm25_model = BM25Okapi(tokenized_documents)

    return embeddings, index, documents, embedding_model, reranker_model, slm_pipeline, bm25_model

COMPANY = "Alphabet Inc"
TICKER = "GOOG"
SEMANTIC_TOP_K: int = 5
SEMANTIC_WEIGHT=0.3
BM25_TOP_K: int = 5
BM25_WEIGHT=0.3
RERANK_TOP_K: int = 5
RERANK_WEIGHT=0.4
SLM_TEMPERATURE: float = 0.2
SLM_MAX_TOKENS: int = 256

# Initialize conversation history for Memory-Augmented Retrieval
conversation_history = []

def basic_retrieve(query, documents, index, embedding_model):
    """Retrieves the most relevant documents using semantic search."""

    query_embedding = embedding_model.encode([query])
    D, I = index.search(query_embedding, SEMANTIC_TOP_K)
    retrieved_docs = [documents[i] for i in I[0]]
    retrieval_scores = D[0].tolist()
    return retrieved_docs, retrieval_scores

def generate_response_basic(query, retrieved_docs, slm_pipeline, temperature=SLM_TEMPERATURE, max_new_tokens=SLM_MAX_TOKENS):
    """
    Generates a response using the SLM and retrieved documents.
    """

    context = "\n\n".join([doc['text'] for doc in retrieved_docs])
    prompt_template = f"""<|system|>
You are a helpful financial assistant. Answer the question based on the context below. Be concise and factual. If the answer is not in the context, say "I cannot answer based on the provided documents."
Context:
{context}</s>
<|user|>
{query}</s>
<|assistant|>
"""
    try:
        response = slm_pipeline(
            prompt_template,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=slm_pipeline.tokenizer.eos_token_id
        )[0]['generated_text']

        response_start = response.find("<|assistant|>") + len("<|assistant|>")
        response_end = response.find("</s>", response_start)
        if response_end == -1:
            response_end = len(response)
        clean_response = response[response_start:response_end].strip()

        return clean_response
    except Exception as e:
        return f"Error generating response: {e}"

def retrieve_with_bm25(query, bm25_model):
    """
    Retrieves documents using BM25 (Best Matching 25).
    BM25 is a lexical retrieval method that ranks documents based on
    term frequency and inverse document frequency.

    Returns document indices and scores
    """

    tokenized_query = nltk.word_tokenize(query.lower())
    bm25_scores = bm25_model.get_scores(tokenized_query)
    top_n_indices = np.argsort(bm25_scores)[::-1][:BM25_TOP_K]
    scores = [bm25_scores[i] for i in top_n_indices]
    return top_n_indices, scores

def hybrid_retrieve_and_rerank(query, documents, index, embedding_model, reranker_model, bm25_model):
    """
    Performs hybrid retrieval (BM25 + Semantic Search) and re-ranks the results, while using the combined scores.

    This function demonstrates a more advanced RAG technique:
    1.  Basic RAG (Semantic Search):
        -   Encodes the query using a Sentence Transformer.
        -   Uses FAISS to perform approximate nearest neighbor search to find
            semantically similar documents.
    2.  Advanced RAG (Hybrid Retrieval and Reranking):
        -   Performs BM25 retrieval to get lexically relevant documents.
        -   Combines the results from BM25 and semantic search.
        -   Reranks the combined results using a Cross-Encoder model, which
            scores the relevance of each document-query pair.

    Returns top-k documents and scores
    """

    # Basic RAG: Semantic Search
    query_embedding = embedding_model.encode([query])
    D_sem, I_sem = index.search(query_embedding, SEMANTIC_TOP_K)
    semantic_ids = list(I_sem[0])
    semantic_scores = list(D_sem[0])

    # Advanced RAG: Hybrid Retrieval (BM25 + Semantic) and Reranking
    # Perform BM25 retrieval
    bm25_ids, bm25_scores = retrieve_with_bm25(query, bm25_model)

    # Combine BM25 and Semantic results
    combined_docs = {}
    combined_scores = {}
    for i, doc_id in enumerate(bm25_ids):
        if doc_id not in combined_docs:
            combined_docs[doc_id] = documents[doc_id]

            # Initialize with BM25 score
            combined_scores[doc_id] = {'bm25': bm25_scores[i], 'semantic': 0.0}
        else:

            # Update BM25 score if document already in combined results
            combined_scores[doc_id]['bm25'] = bm25_scores[i]

    for i, doc_id in enumerate(semantic_ids):
        if doc_id not in combined_docs:
            combined_docs[doc_id] = documents[doc_id]

            # Initialize with semantic score
            combined_scores[doc_id] = {'bm25': 0.0, 'semantic': semantic_scores[i]}
        else:

            # Update semantic score if document already in combined results
            combined_scores[doc_id]['semantic'] = semantic_scores[i]

    # Reranking using Cross-Encoder
    # Create query-document pairs for reranking
    rerank_pairs = [[query, doc['text']] for doc in combined_docs.values()]
    if not rerank_pairs:
        return [], []

    # Predict relevance scores using cross-encoder
    rerank_scores = reranker_model.predict(rerank_pairs)

    # Combine rerank scores with original combined scores using weighted sum
    indexed_combined_scores = []
    doc_ids = list(combined_docs.keys())
    for i, doc_id in enumerate(doc_ids):
        bm25_score = combined_scores[doc_id]['bm25']
        semantic_score = combined_scores[doc_id]['semantic']
        combined_score = (BM25_WEIGHT * bm25_score) + (SEMANTIC_WEIGHT * semantic_score) + (RERANK_WEIGHT * rerank_scores[i])
        indexed_combined_scores.append((doc_id, combined_score))

    # Sort by the combined score
    sorted_combined = sorted(indexed_combined_scores, key=lambda x: x[1], reverse=True)

    # Get top-k document IDs and scores
    top_reranked_ids = [doc_id for doc_id, _ in sorted_combined[:RERANK_TOP_K]]
    top_reranked_docs = [documents[doc_id] for doc_id in top_reranked_ids]
    top_reranked_scores = [score for _, score in sorted_combined[:RERANK_TOP_K]]

    return top_reranked_docs, top_reranked_scores

def generate_response_advanced(query, retrieved_docs, slm_pipeline):
    """
    Generates a response using the SLM, retrieved documents, and conversation history.

    This function also demonstrates a more advanced RAG technique:
    -   Memory-Augmented Retrieval: Uses conversation history to provide context to the SLM.
        This allows the chatbot to maintain context across multiple turns.
    """

    # Memory-Augmented Retrieval: Add conversation history for context
    history_text = ""
    for turn in conversation_history[-3:]:  # Use last 3 turns of conversation history (sliding window)
        history_text += f"<|user|>\n{turn['user']}</s>\n<|assistant|>\n{turn['assistant']}</s>\n"

    # Combine retrieved documents into a single context string
    context = "\n\n".join([doc['text'] for doc in retrieved_docs])
    prompt_template = f"""<|system|>
You are a helpful financial assistant. Answer the question based on the context below and the conversation history. Be concise and factual. If the answer is not in the context or history, say "I cannot answer based on the provided information."
Conversation History:
{history_text}
Context:
{context}</s>
<|user|>
{query}</s>
<|assistant|>
"""
    try:
        response = slm_pipeline(
            prompt_template,
            max_new_tokens=SLM_MAX_TOKENS,
            temperature=SLM_TEMPERATURE,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=slm_pipeline.tokenizer.eos_token_id
        )[0]['generated_text']

        # Extract the assistant's response
        response_start = response.find("<|assistant|>") + len("<|assistant|>")
        response_end = response.find("</s>", response_start)
        if response_end == -1:
            response_end = len(response)
        clean_response = response[response_start:response_end].strip()

        return clean_response
    except Exception as e:
        return f"Error generating response: {e}"

def calculate_confidence(retrieval_scores):
    """
    Calculates a robust confidence score combining the given retrieval scores.

    Returns:
        float: Combined confidence score (0.0 to 1.0).
    """

    if not isinstance(retrieval_scores, (list, np.ndarray)) or (isinstance(retrieval_scores, np.ndarray) and retrieval_scores.size == 0):
        return 0.0

    if isinstance(retrieval_scores, np.ndarray):
        retrieval_scores = np.array(retrieval_scores)

    # Normalize retrieval scores (assuming scores are similarities or distances)
    if np.max(retrieval_scores) - np.min(retrieval_scores) != 0:
        normalized_retrieval_scores = (retrieval_scores - np.min(retrieval_scores)) / (np.max(retrieval_scores) - np.min(retrieval_scores))
    else:
        normalized_retrieval_scores = np.ones_like(retrieval_scores)

    # For simplicity, we'll take the average of raw semantic similarity scores if available
    avg_normalized_retrieval_score = np.mean(normalized_retrieval_scores)
    return avg_normalized_retrieval_score

def is_relevant_query(query):
    """
    Input-side guardrail to check if the query is likely related to finance
    or the specific company.
    """
    query = query.lower()
    finance_keywords = ["revenue", "income", "profit", "loss", "ebitda", "cash flow", "assets",
                        "liabilities", "equity", "financial statement", "10-k", "earnings",
                        "stock", "share", "dividend", "market cap", "valuation", "performance",
                        "quarter", "year", "annual report", "balance sheet", "income statement",
                        "cash flow statement", "operating expenses", "net income", "gross profit"]
    company_keywords = [COMPANY.lower(), TICKER.lower()]

    if any(keyword in query for keyword in finance_keywords) or \
       any(keyword in query for keyword in company_keywords):
        return True
    return False

st.title(f"{COMPANY} ({TICKER}) Financial RAG Chatbot")
st.caption("Answers questions based on the last two years of financial statements.")

if 'conversation' not in st.session_state:
    st.session_state.conversation = []

with st.sidebar:
    st.header("Settings")
    is_advanced_rag = st.toggle("Use Advanced RAG")
    st.caption("Advanced RAG uses Semantic Search + BM25 + Re-ranking + Memory-Augmented Generation.")
    chunk_options = {
        128: "128",
        256: "256",
        512: "512",
        1024: "1024",
    }
    chunk_size = st.radio(
        "Chunk Size",
        options=chunk_options.keys(),
        format_func=lambda option: chunk_options[option],
        index=1,
    )
    if st.button("Clear Conversation"):
        st.session_state.conversation = []
        conversation_history = []

user_query = st.text_input("Ask a financial question:", key="user_query")
search_button = st.button("Ask")

if search_button and user_query:
    if not is_relevant_query(user_query):
        st.warning("This query does not seem relevant to the company's financials. Please ask a relevant question.")
        st.session_state.conversation.append({"user": user_query, "assistant": "This query does not seem relevant to the company's financials.", "confidence": 0.0})
    else:
        with st.spinner("Processing your query..."):
            start_time = time.time()
            embeddings, index, documents, embedding_model, reranker_model, slm_pipeline, bm25_model = load_resources(chunk_size)
            if embeddings is None:
                st.error(f"Could not load resources for chunk size {chunk_size}.  Please ensure the data has been processed for this chunk size.")
                st.stop()
            if not is_advanced_rag:
                retrieved_docs, retrieval_scores = basic_retrieve(user_query, documents, index, embedding_model)
                response = generate_response_basic(user_query, retrieved_docs, slm_pipeline)
            else:
                retrieved_docs, retrieval_scores = hybrid_retrieve_and_rerank(user_query, documents, index, embedding_model, reranker_model, bm25_model)
                response = generate_response_advanced(user_query, retrieved_docs, slm_pipeline)
                # Update conversation history for Memory-Augmented Retrieval
                conversation_history.append({"user": user_query, "assistant": response})

                # Keep only last 5 interactions in conversation history
                if len(conversation_history) > 5:
                    conversation_history.pop(0)

            confidence = calculate_confidence(retrieval_scores)
            end_time = time.time()
            processing_time = end_time - start_time

            # Render the response
            st.session_state.conversation.append({"user": user_query, "assistant": response, "confidence": confidence, "chunk_size": chunk_size, "processing_time": processing_time})

st.markdown("---")
st.subheader("Conversation History")
for turn in st.session_state.conversation:
    st.markdown(f"**You:** {turn['user']}")
    st.markdown(f"**Assistant:** {turn['assistant']}")
    st.markdown(f"**Confidence:** {turn['confidence']:.2f}")
    st.markdown(f"**Chunk Size:** {turn.get('chunk_size', 'N/A')}")
    st.markdown(f"**Processing Time:** {turn.get('processing_time', 'N/A')} seconds")
    st.markdown("---")
