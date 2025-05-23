import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
from src.config import Config
from src.data_processing import load_df, create_unique_questions_df
from src.embedding import generate_embeddings
from src.faiss_index_search import build_faiss_index
import faiss


def setup_search_system(dataset_input_path=None,
                        questions_df_output_path=None,
                        emb_model_name_or_path=None,
                        embedding_output_path=None,
                        faiss_index_output_path=None,
                        sample_size=None):
    """
    Set up the search system by processing data, generating embeddings, and building index.

    Args:
        dataset_input_path: Path to the dataset
        questions_df_output_path: Path to save questions DataFrame
        emb_model_name_or_path: Name or path of the embedding model
        embedding_output_path: Path to save embeddings
        faiss_index_output_path: Path to save FAISS index
        sample_size: Number of samples to use (for testing)
    """
    # Use config values as defaults
    dataset_input_path = dataset_input_path or Config.DATA_PATH
    questions_df_output_path = questions_df_output_path or Config.QUESTIONS_DF_PATH
    emb_model_name_or_path = emb_model_name_or_path or Config.EMBEDDING_MODEL
    embedding_output_path = embedding_output_path or Config.EMBEDDINGS_PATH
    faiss_index_output_path = faiss_index_output_path or Config.FAISS_INDEX_PATH

    try:
        # Load dataset
        dataset = load_df(dataset_input_path)
        questions_df, unique_questions = create_unique_questions_df(dataset)

        # Sample if needed
        if sample_size and len(questions_df) > sample_size:
            questions_df = questions_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            unique_questions = questions_df['clean_question'].tolist()
            print(f"Sampled {sample_size} questions for testing")

        questions_df.to_pickle(questions_df_output_path)

        # Generate and save embeddings
        emb_list = generate_embeddings(unique_questions, emb_model_name_or_path)

        print(f"Generated embeddings shape: {emb_list.shape}")
        np.save(embedding_output_path, emb_list)
        print(f"Embeddings saved to {embedding_output_path}")

        # Build and save index
        faiss_index = build_faiss_index(emb_list)
        faiss.write_index(faiss_index, faiss_index_output_path)
        print(f"FAISS index saved to : {faiss_index_output_path}")
    except Exception as e:
        print(f"Error in setup_search_system: {e}")
        raise


if __name__ == "__main__":
    setup_search_system()
