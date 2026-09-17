import argparse
import json
from lib.multimodal_search import MultiModalSearch,search_image_command



def main() -> None:
    
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    verify_image_embedding_parser = subparsers.add_parser("verify_image_embedding", help="Verify the image embedding model")
    image_search_parser = subparsers.add_parser("image_search", help="Search for documents based on an image")
    
    verify_image_embedding_parser.add_argument("image_path", type=str, help="Path to the image file to verify embedding")
    image_search_parser.add_argument("image_path", type=str, help="Path to the image file to search with")
    args = parser.parse_args()
    
    match args.command:
        case "verify_image_embedding":
            image_path = args.image_path
            multimodal_search = MultiModalSearch()
            multimodal_search.verify_image_embedding(image_path)
        case "image_search":
            image_path = args.image_path
            results = search_image_command(image_path)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']} (similarity: {result['similarity']:.3f})")
                print(f"   {result['description'][:200]}...")
        
        case _:
            parser.print_help()
        
if __name__ == "__main__":
    main()