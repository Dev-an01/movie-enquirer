from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.search_utils import load_movies
class MultiModalSearch:
    
    def __init__(self, document: list, model_name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = document
        self.texts = [doc["title"] + ": " + doc["description"] for doc in document]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=False)
        
    def verify_image_embedding(self, image_path: str) -> None:
        try:
            image = Image.open(image_path)
            embedding = self.model.encode(image)
            print(f"Embedding shape: {embedding.shape[0]} dimensions")
        except Exception as e:
            print(f"Error generating image embedding: {e}")
            
    def search_with_image(self, image_path: str, limit: int = 5) -> list[dict]:
        try:
            image = Image.open(image_path)
            image_embedding = self.model.encode(image)
        except Exception as e:
            print(f"Error generating image embedding: {e}")
            return []
            
        results = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity = self.model.similarity(image_embedding, text_embedding)
            results.append({
                "id": self.documents[i]["id"],
                "title": self.documents[i]["title"],
                "description": self.documents[i]["description"],
                "similarity": similarity.item(),
            })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = results[:limit]
        return top_results
        
def search_image_command(image_path: str, limit: int = 5) -> list[dict]:
    documents = load_movies()
    multimodal_search = MultiModalSearch(documents)
    results = multimodal_search.search_with_image(image_path, limit=limit)
    return results