import numpy as np
from sentence_transformers import CrossEncoder
from .config import Config
from .utils import clean_question_text


def load_reranker(model_name=None):
    """
    Load cross-encoder model for reranking.

    Args:
        model_name: Name or path of the model. If None, uses default from config.

    Returns:
        Loaded CrossEncoder model
    """
    model_name = model_name or Config.RERANKER_MODEL
    print(f"Loading reranker model: {model_name}")
    return CrossEncoder(model_name)


def rerank_candidates(query_text, candidates, reranker):
    """
    Rerank candidates using cross-encoder.

    Args:
        query_text: Query text
        candidates: List of candidate questions with metadata
        reranker: CrossEncoder model

    Returns:
        List of reranked candidates
    """
    clean_query = clean_question_text(query_text)

    # Prepare query-candidate pairs
    query_pairs = [(clean_query, candidate['clean_question']) for candidate in candidates]

    # Get reranking scores
    scores = reranker.predict(query_pairs)

    # Add scores to candidates
    for i, score in enumerate(scores):
        probability = 1 / (1 + np.exp(-score))
        candidates[i]['rerank_score'] = float(score)
        candidates[i]['similarity_percentage'] = f"{probability * 100:.1f}%"

    # Sort by rerank score
    reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

    # Add ranks
    for i, result in enumerate(reranked):
        result['rank'] = i + 1

    return reranked
