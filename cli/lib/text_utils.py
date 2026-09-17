import string
from nltk.stem import PorterStemmer
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    stemmed_tokens = []
    for token in tokens:
        stemmed_token = stemmer.stem(token)
        stemmed_tokens.append(stemmed_token)
    return stemmed_tokens

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    valid_tokens = remove_stopwords(valid_tokens, load_stopwords())
    valid_tokens = stem_tokens(valid_tokens)
    return valid_tokens

def tokenize_single_term(term: str) -> str:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError("term must be a single token")
    return tokens[0]

def tf_tokenize(text: str) -> str:
    tokens = tokenize_text(text)
    if len(tokens) != 1:
        raise Exception("Input text must contain exactly one token after preprocessing.")
    return tokens[0]

def remove_stopwords(tokens: list[str], stopwords: set[str]) -> list[str]:
    valid_tokens = []
    for token in tokens:
        if token not in stopwords:
            valid_tokens.append(token)
    return valid_tokens
