import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import os

# Load data and models with correct paths
documents_path = os.path.join(os.path.dirname(__file__), '../data/documents.txt')
index_path = os.path.join(os.path.dirname(__file__), '../data/faiss_index.index')

documents = [doc.strip() for doc in open(documents_path).readlines()]
index = faiss.read_index(index_path)
model = SentenceTransformer('all-MiniLM-L12-v2')
generator = pipeline('text-generation', model='gpt2')

def retrieve_documents(query, index, model, documents, top_k=3):
    query_embedding = model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding), top_k)
    retrieved_documents = [documents[idx] for idx in indices[0]]
    return retrieved_documents

def generate_response(retrieved_docs, query, generator):
    context = " ".join(retrieved_docs)
    input_text = f"Context: {context}\n\nQuery: {query}\n\nAnswer:"
    response = generator(input_text, max_length=100, do_sample=True)
    return response[0]['generated_text']

st.title("Question and Answer with RAG")

query = st.text_input("Enter your question:")

if st.button("Submit"):
    if query:
        retrieved_docs = retrieve_documents(query, index, model, documents)
        response = generate_response(retrieved_docs, query, generator)
        st.write("**Response:**")
        st.write(response)
        st.write("**Retrieved Documents:**")
        for doc in retrieved_docs:
            st.write(f"- {doc}")
    else:
        st.write("Please enter a question.")
