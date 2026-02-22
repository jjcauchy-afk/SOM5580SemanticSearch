import streamlit as st
from sentence_transformers import SentenceTransformer, util

# 1. Page Config & Title
st.set_page_config(page_title="Semantic Search 4", page_icon="🔍")
st.title("🔍 Semantic Search App")

# 2. Load Model (Cached so it only loads once)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# 3. Define Corpus
corpus = [
    "The capital of France is Paris.",
    "A man is eating a piece of bread.",
    "Python is a popular programming language.",
    "The new smartphone has a great camera.",
    "London is the largest city in the UK."
]

# 4. Pre-encode Corpus
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# 5. User Input
query = st.text_input("Enter your search query:", "Which city is the France's capital?")

if query:
    # 6. Encode Query & Search
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)

    # 7. Display Results
    st.subheader("Top Result:")
    for hit in hits[0]:
        score = hit['score']
        result_text = corpus[hit['corpus_id']]
        
        st.success(f"**Result:** {result_text}")
        st.info(f"**Similarity Score:** {score:.4f}")
