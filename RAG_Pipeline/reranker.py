from sentence_transformers import CrossEncoder
from RAG_Pipeline.retriever import query_corrected_retrieval

reranker = CrossEncoder('BAAI/bge-reranker-large')

def reranked_retriever(query, top_k=5):
    docs, images = query_corrected_retrieval(query, top_k=20)
    
    # If similarity scores<0.3 for both text and images, return early
    if docs == ["No relevant text found."] and images == ["No relevant images found."]:
        return "No relevant text found.", "No relevant images found."
    
    # Create pairs for the CrossEncoder
    pairs = [[query, d] for d in docs]
    scores = reranker.predict(pairs)
    
    # Rank documents based on scores
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    
    # Initialize and build the context string
    text_context = ""
    for doc in ranked[0:top_k]:
        text_context += doc[0] + "\n\n"
        
    return text_context, images

#images are returned in the following format:
# List[Dict]->[
#     { "image": "base64_string", "image_path": "path_to_image" },
#     ]
