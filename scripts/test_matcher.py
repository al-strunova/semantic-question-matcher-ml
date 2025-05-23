import pandas as pd
from src.config import Config
from src.embedding import load_embedding_model
from src.faiss_index_search import load_faiss_index
from src.reranker import load_reranker
from src.question_matcher import search_similar_questions


def init_search_system(questions_df_path=None,
                       faiss_index_path=None,
                       emb_model_name=None,
                       reranker_name=None):
    """
    Initialize the search system for testing.

    Args:
        questions_df_path: Path to questions DataFrame
        faiss_index_path: Path to FAISS index
        emb_model_name: Name or path of the embedding model
        reranker_name: Name or path of the reranker model

    Returns:
        Tuple of (faiss_index, questions_df, emb_model, reranker)
    """
    # Use config values as defaults
    questions_df_path = questions_df_path or Config.QUESTIONS_DF_PATH
    faiss_index_path = faiss_index_path or Config.FAISS_INDEX_PATH
    emb_model_name = emb_model_name or Config.EMBEDDING_MODEL
    reranker_name = reranker_name or Config.RERANKER_MODEL

    print("Initializing search system...")

    # Load components
    faiss_index = load_faiss_index(faiss_index_path)
    questions_df = pd.read_pickle(questions_df_path)
    emb_model = load_embedding_model(emb_model_name)
    reranker = load_reranker(reranker_name)

    print("Search system initialized!")
    return faiss_index, questions_df, emb_model, reranker


# Example usage
if __name__ == "__main__":
    # Initialize the system
    faiss_index, questions_df, emb_model, reranker = init_search_system()

    # Test a query
    test_query = "What is machine learning?"
    results = search_similar_questions(test_query, emb_model, faiss_index, questions_df, reranker)

    print("\nSearch results:")
    for result in results[:3]:
        print(f"{result['rank']}. {result['clean_question']} ({result['similarity_percentage']})")
