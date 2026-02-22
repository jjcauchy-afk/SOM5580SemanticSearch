import streamlit as st
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------- Page Configuration --------------------------
st.set_page_config(
    page_title = "ISOM5580 Semantic Search",
    page_icon = "🔍",
    layout = "wide"
)

# -------------------------- Cached Functions --------------------------
@st.cache_resource(show_spinner = "Loading embedding model...")
def load_embedding_model():
    """Load the SentenceTransformer model (cached)"""
    try:
        # Lightweight model, balanced for speed and performance
        #model = SentenceTransformer('all-MiniLM-L6-v2')
        model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

@st.cache_data(show_spinner = "Processing text and generating embeddings...")
def process_text_and_generate_embeddings(text, _model):
    """Chunk text and generate semantic vectors"""
    # 1. Chunking: Split by paragraphs, filter empty lines
    chunks = [chunk.strip() for chunk in re.split(r'\n\s*\n', text) if chunk.strip()]

    # Fallback: if too few paragraphs, split by sentence endings
    if len(chunks) < 3:
        chunks = [sent.strip() for sent in re.split(r'(?<=[.!?])', text) if sent.strip()]

    # 2. Generate embeddings (using _model to skip hashing)
    embeddings = _model.encode(chunks, convert_to_numpy=True)

    return chunks, embeddings


# -------------------------- Core Search Logic --------------------------
def semantic_search(query, chunks, embeddings, model, top_k=5):
    """
    Semantic search core logic
    """
    # Generate embedding for the user query
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Sort by similarity and get top_k results
    sorted_indices = similarities.argsort()[::-1][:top_k]
    results = [
        {
            "text": chunks[idx],
            "similarity": round(similarities[idx] * 100, 2)
        }
        for idx in sorted_indices
    ]

    return results


# -------------------------- Main UI Logic --------------------------
def main():
    st.title("🔍 Semantic Search App")
    st.divider()

    # Load model
    model = load_embedding_model()

    # Sidebar: Data Source Settings
    st.subheader("Step 1. Prepare Data Source")
    # Option 1: File Upload
    uploaded_file = st.file_uploader(
        "Option 1: Upload Text File (.txt)",
        type = ["txt"],
        help = "Supports plain text files. UTF-8 encoding is recommended."
    )

    # Option 2: Manual Input
    manual_text = st.text_area(
        "Option 2: Paste Text Directly",
        height = 200,
        placeholder = "Paste your content here..."
    )

    # Parameters
    st.subheader("Step 2. Select Number of Results")
    col_slider, _ = st.columns([1, 1]) 
    with col_slider:
        top_k = st.slider("Number of Results", min_value=1, max_value=10, value=5)

    # Handle Data Source
    text_source = None
    if uploaded_file is not None:
        try:
            text_source = uploaded_file.read().decode("utf-8")
            st.success(f"✅ Loaded: {uploaded_file.name}")
        except UnicodeDecodeError:
            # Fallback for other encodings
            text_source = uploaded_file.read().decode("latin-1")
            st.success(f"✅ Loaded: {uploaded_file.name} (Latin-1)")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    elif manual_text:
        text_source = manual_text
        st.success("✅ Using manually entered text")

    # Search Section
    st.subheader("Step 3. Start Searching")
    query = st.text_input("Enter search keywords or sentences", placeholder="e.g., Application scenarios of AI...")

    # UI Guidance
    if not text_source:
        st.info("💡 Please upload a file or enter text in the sidebar to begin.")
    elif text_source and not query:
        st.info("💡 Enter a query and click 'Search' to see semantic matches.")

    search_btn = st.button("Search", type="primary", disabled=not (text_source and query))

    # Execution
    if search_btn and text_source and query:
        with st.spinner("Finding best matches..."):
            # Process text and embeddings
            chunks, embeddings = process_text_and_generate_embeddings(text_source, model)

            # Perform search
            results = semantic_search(query, chunks, embeddings, model, top_k)

            # Display Results
            st.divider()
            st.subheader(f"🔎 Search Results (Top {top_k})")
            for idx, result in enumerate(results, 1):
                with st.expander(f"Result {idx} (Similarity: {result['similarity']}%)"):
                    st.write(result["text"])


if __name__ == "__main__":
    main()
