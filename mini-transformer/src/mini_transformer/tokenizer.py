import re

def tokenize(text):
    # 1. Convert text to lowercase
    text = text.lower()

    # 2. Separate punctuation from words
    text = re.sub(r'([.,!?;:()"\'])', r' \1 ', text)

    # 3. Split into tokens
    tokens = text.split()

    return tokens