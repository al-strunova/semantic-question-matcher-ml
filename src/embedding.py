from sentence_transformers import SentenceTransformer
from .utils import clean_question_text
from .config import Config


def load_embedding_model(model_name=None):
    """
    Load sentence transformer model for embeddings.

    Args:
        model_name: Name or path of the model. If None, uses default from config.

    Returns:
        Loaded SentenceTransformer model
    """
    model_name = model_name or Config.EMBEDDING_MODEL
    embed_model = SentenceTransformer(model_name)
    return embed_model


def generate_embeddings(texts_to_embed_list: list, model_name=None):
    """
    Generate embeddings for a list of texts.

    Args:
        texts_to_embed_list: List of texts to embed
        model_name: Name or path of the model. If None, uses default from config.

    Returns:
        Numpy array of embeddings
    """
    print("Starting embedding generation...")
    model_name = model_name or Config.EMBEDDING_MODEL

    embed_model = load_embedding_model(model_name)
    embed_list = embed_model.encode(texts_to_embed_list,
                                    batch_size=Config.BATCH_SIZE,
                                    show_progress_bar=True,
                                    convert_to_numpy=True
                                    )

    print(f"Generated embeddings shape: {embed_list.shape}")
    return embed_list


def embed_single_question(model, question):
    """
    Generate embedding for a single question.

    Args:
        model: SentenceTransformer model
        question: Question text

    Returns:
        Embedding vector for the question
    """
    clean_question = clean_question_text(question)
    emb_question = model.encode([clean_question])
    return emb_question[0]
