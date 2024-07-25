import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def read_documents(file_path):
    with open(file_path, 'r') as file:
        documents = file.readlines()
    return [doc.strip() for doc in documents]

def create_faiss_index(documents, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    document_embeddings = model.encode(documents, convert_to_tensor=False)
    
    dimension = document_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(document_embeddings))
    
    return index, document_embeddings

def save_faiss_index(index, file_path):
    faiss.write_index(index, file_path)

if __name__ == "__main__":
    # Ensure the correct relative path
    file_path = os.path.join(os.path.dirname(__file__), '../data/documents.txt')
    documents = read_documents(file_path)
    index, embeddings = create_faiss_index(documents)
    
    # Save the index and embeddings to the data directory
    save_faiss_index(index, os.path.join(os.path.dirname(__file__), '../data/faiss_index.index'))
    np.save(os.path.join(os.path.dirname(__file__), '../data/document_embeddings.npy'), embeddings)
