from .config import Config
from .faiss_index_search import search_faiss
from .reranker import rerank_candidates


def search_similar_questions(query_text, emb_model, index, questions_df, reranker,
                             k_candidates=None, k_final=None):
    """
    Search for questions similar to the input query.

    Args:
        query_text: Query text
        emb_model: SentenceTransformer model
        index: FAISS index
        questions_df: DataFrame with questions
        reranker: CrossEncoder model
        k_candidates: Number of candidates to retrieve from FAISS
        k_final: Number of final results to return after reranking

    Returns:
        List of dictionaries with similar questions and metadata
    """
    k_candidates = k_candidates or Config.DEFAULT_CANDIDATES
    k_final = k_final or Config.DEFAULT_RESULTS

    # Step 1: Get candidates from FAISS
    distances, indices = search_faiss(query_text, emb_model, index, k_candidates)

    # Step 2: Prepare candidates for reranking
    candidates = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        candidates.append({
            'question_id': questions_df.iloc[idx]['qid'],
            'clean_question': questions_df.iloc[idx]['clean_question'],
            'faiss_rank': i + 1,
            'faiss_distance': float(distance)
        })

    # Step 3: Rerank candidates
    reranked_results = rerank_candidates(query_text, candidates, reranker)

    # Step 4: Return top k_final results
    return reranked_results[:k_final]
