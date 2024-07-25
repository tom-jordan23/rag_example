import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os

def load_faiss_index(file_path):
    return faiss.read_index(file_path)

def retrieve_documents(query, index, model, documents, top_k=3):
    query_embedding = model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding), top_k)
    retrieved_documents = [documents[idx] for idx in indices[0]]
    return retrieved_documents

def generate_response(retrieved_docs, query, generator):
    context = " ".join(retrieved_docs)
    input_text = f"Context: {context}\nQuery: {query}\nAnswer:"
    response = generator(input_text, max_length=100, do_sample=True)
    return response[0]['generated_text']

if __name__ == "__main__":
    documents_path = os.path.join(os.path.dirname(__file__), '../data/documents.txt')
    index_path = os.path.join(os.path.dirname(__file__), '../data/faiss_index.index')
    embeddings_path = os.path.join(os.path.dirname(__file__), '../data/document_embeddings.npy')
    
    documents = [doc.strip() for doc in open(documents_path).readlines()]
    index = load_faiss_index(index_path)
    embeddings = np.load(embeddings_path)
    
    query = "Where is the Eiffel Tower located?"
    model = SentenceTransformer('all-MiniLM-L6-v2')
    retrieved_docs = retrieve_documents(query, index, model, documents)
    
    generator = pipeline('text-generation', model='gpt2')
    response = generate_response(retrieved_docs, query, generator)
    print(response)
