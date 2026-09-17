from .text_utils import tokenize_text, tokenize_single_term
from .search_utils import load_movies, load_stopwords
import pickle
import math
import os 

BM25_K1 = 1.5
BM25_B = 0.75
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "cache")

class InvertedIndex:
    
    # def __init__(self, index: dict[str,list[int]] = None, docmap: dict[int,str] = None):
    #     self.index = {}
    
    def __init__(self):
        self.index = {}
        self.doc_lengths = {}
        self.docmap= {}
        self.term_frequencies = {}
        self.movies = load_movies()
        # how to fix this error ? "CACHE_DIR" is not defined. Ans-> import os and define CACHE_DIR
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
    # what is this error ?
   # "docmap": Unknown word.cSpell
#(variable) docmap: dict" :   Ans-> 
    
    def load(self):
        try:
            with open("cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)
            with open("cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)
            with open("cache/term_frequencies.pkl", "rb") as f:
                self.term_frequencies = pickle.load(f)
            with open("cache/doc_lengths.pkl", "rb") as f:
                self.doc_lengths = pickle.load(f)
            
        except FileNotFoundError:
            raise FileNotFoundError("Index or docmap file not found. Please build the index first.")
        
    def  __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = {}  
            if token not in self.term_frequencies[doc_id]:
                self.term_frequencies[doc_id][token] = 1
            else: 
                self.term_frequencies[doc_id][token] += 1
            
    def  __get_avg_doc_length(self) -> float:
        sum_lengths = sum(self.doc_lengths.values())
        num_docs = len(self.doc_lengths)
        return sum_lengths / num_docs if num_docs > 0 else 0.0

    def get_document(self, term:str) -> list[int]:
        return self.index[term]
    
    def get_tf(self, doc_id: int, term: str) -> int:
        if doc_id in self.term_frequencies and term in self.term_frequencies[doc_id]:
            return self.term_frequencies[doc_id][term]
        return 0
    
    def get_idf(self, term: str) -> float:
        total_docs = len(self.docmap)
        if term in self.index:
            matching_docs = len(self.index[term])
            return math.log(total_docs / (1 + matching_docs))
        return 0.0

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
    def get_bm25_idf(self, term: str) -> float:
        total_docs = len(self.docmap)
        if term in self.index:
            matching_docs = len(self.index[term])
            #upto 2decimal places
            ans = math.log((total_docs - matching_docs + 0.5) / (matching_docs + 0.5) + 1)
            return ans
            #return math.log((total_docs - matching_docs + 0.5) / (matching_docs + 0.5) + 1)
        return 0.0
    
    def get_bm25_tf(self, doc_id: int, term: str, k1 : int = BM25_K1, b : float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        length_norm = (1 - b) + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        tf_component = (tf *(k1+1)) / (tf + k1 * length_norm)
        
        return tf_component

    def bm25(self, doc_id: int, term : str) -> float:
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        if tf is None or idf is None:
            return 0.0
        bm25_score = tf * idf
        return bm25_score
        

    def bm25_search(self, query: str, limit : int) -> list[tuple[int, float]]:
        tokens = tokenize_text(query)
        scores = {}
        for doc_id in self.docmap.keys():
            score = 0.0
            for token in tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:limit]

    def build(self):
        movies = load_movies()
        for i, movie in enumerate(movies):
            self.docmap[i+1] = movie["title"]
            self.__add_document(i+1, f"{movie["title"]} {movie["description"]}")
            
    def save(self):
        #saving as pickle file
        #wb = 
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open("cache/doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths, f)
        
        
    
def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()
    
def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(tokenize_single_term(term))


def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id, tokenize_single_term(term))

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(tokenize_single_term(term))

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, tokenize_single_term(term))
#how to take optional argument in command line
def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b : float = BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, tokenize_single_term(term))
    
    