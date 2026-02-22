import streamlit as st
# 1. Install dependencies: pip install sentence-transformers
from sentence_transformers import SentenceTransformer, util

# 3. Define your corpus (the documents to search through)
corpus = [
    "The capital of France is Paris.",
    "A man is eating a piece of bread.",
    "Python is a popular programming language.",
    "The new smartphone has a great camera.",
    "London is the largest city in the UK."
]

# 4. Encode the corpus into embeddings
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# 5. Define a query
#query = "Which programming language is popula?"
query = "Which city is the Lodon capital?"

# 6. Encode the query and find similarity
query_embedding = model.encode(query, convert_to_tensor=True)

# Use util.semantic_search for a quick top-k retrieval
hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)

# 7. Print the most similar result
for hit in hits[0]:
    print(f"Result: {corpus[hit['corpus_id']]} (Score: {hit['score']:.4f})")
