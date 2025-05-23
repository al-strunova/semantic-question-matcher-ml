import pandas as pd
from typing import Tuple, List
from .config import Config
from .utils import clean_question_text


def load_df(file_name: str = None) -> pd.DataFrame:
    """
    Load and process the Quora Question Pairs dataset.

    Args:
        file_name: Path to the dataset file. If None, uses default from config.

    Returns:
        DataFrame with processed question pairs
    """
    file_name = file_name or Config.DATA_PATH
    print(f"Loading '{file_name}' dataset...")

    try:
        df = pd.read_csv(file_name,
                         sep='\t',
                         on_bad_lines='skip',
                         dtype=object,
                         quoting=3)
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_name}")
        raise

    essential_columns = ['qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
    df_clean = df.dropna(subset=essential_columns, how='any').reset_index(drop=True)

    processed_df = pd.DataFrame({
        'id_left': df_clean['qid1'],
        'id_right': df_clean['qid2'],
        'text_left': df_clean['question1'],
        'text_right': df_clean['question2'],
        'label': df_clean['is_duplicate'].astype(int)
    })

    print(f"Successfully loaded and processed '{file_name}'. Shape of processed_df: {processed_df.shape}")
    return processed_df


def create_unique_questions_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create a DataFrame with unique questions from question pairs.

    Args:
        df: DataFrame with question pairs

    Returns:
        Tuple of (DataFrame with unique questions, list of clean question texts)
    """
    print("Starting question_df creation...")

    # Extract left questions and apply cleaning directly
    left_questions = df[['id_left', 'text_left']].rename(columns={'id_left': 'qid', 'text_left': 'clean_question'})
    left_questions['clean_question'] = left_questions['clean_question'].apply(clean_question_text)

    # Extract right questions and apply cleaning directly
    right_questions = df[['id_right', 'text_right']].rename(columns={'id_right': 'qid', 'text_right': 'clean_question'})
    right_questions['clean_question'] = right_questions['clean_question'].apply(clean_question_text)

    # Combine and remove duplicates by qid first
    combined_df = (pd.concat([left_questions, right_questions], ignore_index=True)
                   .drop_duplicates(subset=['qid'], keep='first'))

    # Remove duplicates by clean question text (final deduplication)
    all_questions_df = combined_df.drop_duplicates(subset=['clean_question'], keep='first').reset_index(drop=True)
    texts_to_embed_list = all_questions_df['clean_question'].tolist()

    print(f"Successfully created question_df. Shape of question_df: {all_questions_df.shape}")
    return all_questions_df, texts_to_embed_list
