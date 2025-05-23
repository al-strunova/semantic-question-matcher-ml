class Config:
    """Configuration settings for the similar questions system."""

    # Model paths
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # File paths
    DATA_PATH = "data/qqp/train.tsv"
    QUESTIONS_DF_PATH = "models/questions_df.pkl"
    EMBEDDINGS_PATH = "models/questions_embeddings.npy"
    FAISS_INDEX_PATH = "models/questions_index.faiss"

    # Search parameters
    DEFAULT_CANDIDATES = 50
    DEFAULT_RESULTS = 10
    BATCH_SIZE = 64