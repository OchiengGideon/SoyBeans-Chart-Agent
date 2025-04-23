import os
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb import Client
from chromadb.config import Settings
import chromadb.config

# 1. Load multilingual embedding model
EMBEDDING_MODEL_NAME = "distiluse-base-multilingual-cased-v2"
emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# 2. Load and chunk PDF text
def load_and_chunk_pdfs(folder_path, chunk_size=500, overlap=100):
    text = ""
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".pdf"):
            doc = fitz.open(os.path.join(folder_path, fname))
            for page in doc:
                text += page.get_text()
    words = text.split()
    chunks = [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size - overlap)
    ]
    return [c for c in chunks if len(c) > 100]

# 3. Build or load FAISS store
def get_or_create_faiss(chunks, embeddings_model, index_path="faiss_index"):
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings_model, allow_dangerous_deserialization=True)
    else:
        faiss_store = FAISS.from_texts(chunks, embeddings_model)
        faiss_store.save_local(index_path)
        return faiss_store

# 4. Initialize Chroma client (persistent)
chroma_client = Client(chromadb.config.Settings())

# 5. Create or get Chroma collection and store data if needed
def setup_chroma_collection(chunks, name="soybeans"):
    collection = chroma_client.get_or_create_collection(name)
    existing_ids = set(collection.peek()["ids"])
    new_chunks = [c for i, c in enumerate(chunks) if f"chunk_{i}" not in existing_ids]

    if new_chunks:
        new_embeddings = emb_model.encode(new_chunks).tolist()
        new_ids = [f"chunk_{i}" for i, c in enumerate(chunks) if f"chunk_{i}" not in existing_ids]
        collection.add(
            documents=new_chunks,
            embeddings=new_embeddings,
            ids=new_ids
        )
    return collection

# 6. VectorStore with local + Chroma fallback
class VectorStore:
    def __init__(self, local_store=None, chroma_collection=None):
        self.local = local_store
        self.chroma = chroma_collection

    def search(self, query, top_k=3):
        if self.local:
            local_results = self.local.similarity_search(query, k=top_k)
            if local_results:
                return [r.page_content for r in local_results]
        q_emb = emb_model.encode([query], convert_to_tensor=True)
        response = self.chroma.query(
            query_embeddings=q_emb.cpu().numpy().tolist(),
            n_results=top_k
        )
        return [doc for doc in response["documents"][0]]

# --- Startup ---
if __name__ == "__main__":
    chunks = load_and_chunk_pdfs("pdfs")
    hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    local_store = get_or_create_faiss(chunks, hf_embeddings)
    chroma_collection = setup_chroma_collection(chunks)

    VS = VectorStore(local_store, chroma_collection)

    # Example search
    results = VS.search("What is the market for soybeans in Africa?")
    for res in results:
        print("\n---\n", res)
