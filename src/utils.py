import re


def clean_question_text(raw_str: str) -> str:
    """
    Clean question text by lowercasing, removing extra whitespace and trimming.

    Args:
        raw_str: The raw question text to clean

    Returns:
        Cleaned text string
    """
    text = str(raw_str).lower()  # Ensure input is string and lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace one or more whitespace characters with a single space
    text = text.strip()  # Remove leading/trailing spaces
    return text
