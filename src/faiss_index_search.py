from .embedding import embed_single_question
from .config import Config
import faiss


def build_faiss_index(emb_list):
    """
    Build FAISS index from embeddings.

    Args:
        emb_list: Numpy array of embeddings

    Returns:
        FAISS index
    """
    print("Building FAISS index...")
    dimension = emb_list.shape[1]
    index = faiss.IndexFlatL2(dimension)

    print("Adding embeddings to index...")
    index.add(emb_list.astype('float32'))

    print(f"Successfully created index with {index.ntotal} vectors")
    return index


def load_faiss_index(faiss_index_output_path=None):
    """
    Load FAISS index from file.

    Args:
        faiss_index_output_path: Path to the index file. If None, uses default from config.

    Returns:
        FAISS index
    """
    path = faiss_index_output_path or Config.FAISS_INDEX_PATH
    return faiss.read_index(path)


def search_faiss(query_text, emb_model, index, k_candidates=None):
    """
    Search for similar questions using FAISS.

    Args:
        query_text: Query text
        emb_model: SentenceTransformer model
        index: FAISS index
        k_candidates: Number of candidates to retrieve

    Returns:
        Tuple of (distances, indices)
    """
    k_candidates = k_candidates or Config.DEFAULT_CANDIDATES

    emb_question = embed_single_question(emb_model, query_text)
    emb_question = emb_question.astype('float32').reshape(1, -1)
    distances, indices = index.search(emb_question, k=k_candidates)

    return distances, indices
