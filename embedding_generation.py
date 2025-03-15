import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def generate_embeddings(chunks_file, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Generates embeddings for the given chunks and saves them along with a FAISS index."""
    with open(chunks_file, "r") as f:
        documents = json.load(f)

    embedding_model = SentenceTransformer(embedding_model_name)
    embeddings = embedding_model.encode([doc["text"] for doc in documents])
    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap(index)
    index.add_with_ids(embeddings, np.array(range(len(documents))))

    # Save embeddings and index
    np.save(f"embeddings_{chunk_size}.npy", embeddings)
    faiss.write_index(index, f"embeddings_{chunk_size}.index")
    print(f"Embeddings and index saved to embeddings_{chunk_size}.npy and embeddings_{chunk_size}.index")

if __name__ == "__main__":
    chunk_sizes = [128, 256, 512, 1024]

    for chunk_size in chunk_sizes:
        output_file = f"chunks_{chunk_size}.json"
        generate_embeddings(output_file)
