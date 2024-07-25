# RAG Sample Project

This is a minimalistic sample project demonstrating how to create and use a local vector database for Retrieval-Augmented Generation (RAG) with a HuggingFace model using PyTorch.

## Setup

1. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

2. Prepare your documents in `data/documents.txt`.

3. Create the FAISS vector database:

    ```bash
    python scripts/create_vector_db.py
    ```

4. Run the Streamlit app:

    ```bash
    streamlit run app/app.py
    ```

## Project Structure

- `data/documents.txt`: Sample documents.
- `scripts/create_vector_db.py`: Script to create FAISS vector database.
- `scripts/retrieve_and_generate.py`: Script to retrieve documents and generate responses.
- `app/app.py`: Streamlit app for the frontend.
- `requirements.txt`: Required Python packages.
- `README.md`: Project documentation.
