#importing all the dependencies
import google.generativeai as genai
import os
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = "models/text-embedding-004" # Google's latest embedding model    
def get_document_embeddings(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using Google's text-embedding-004 model."""
    genai.configure(api_key=GOOGLE_API_KEY)
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=texts,
        task_type="retrieval_document", # optimized for storage/retrieval
    )
    return result["embedding"]

def get_query_embedding(query: str) -> list[float]:
    """Embed a single query string."""
    genai.configure(api_key=GOOGLE_API_KEY)
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query", # optimized for query retrieval
    )
    return result["embedding"]

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(image_path):
    image = Image.open(image_path)
    inputs = clip_processor(images=image, return_tensors="pt")
    outputs = clip_model.get_image_features(**inputs)
    return outputs.detach().numpy()[0].tolist()
def embed_image_query(query:str,):
    clip_inputs = clip_processor(text=[query], return_tensors="pt", padding=True, truncation=True)
    clip_vector = clip_model.get_text_features(**clip_inputs).detach().numpy()[0].tolist()

    return clip_vector