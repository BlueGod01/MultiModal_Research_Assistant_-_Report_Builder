from pinecone import Pinecone, ServerlessSpec
import os
import logging
from Data_Ingestion_Pipeline.embedder import get_document_embeddings, get_query_embedding
from Data_Ingestion_Pipeline.embedder import embed_image, embed_image_query
import hashlib
from typing import Optional, List
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "multimodal-documents"
EMBEDDING_MODEL = "models/text-embedding-004" # Google's latest embedding model
EMBEDDING_DIMENSION = 768 # text-embedding-004 outputs 768-dim vectors
BATCH_SIZE = 96 # Pinecone upsert batch size

def init_pinecone_index()-> Pinecone.Index:
    """Create or connect to a Pinecone serverless index."""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)

        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logging.info(f"Created Pinecone index: {PINECONE_INDEX_NAME}")

        return pc.Index(PINECONE_INDEX_NAME)
    
    except Exception as e:
        logging.error(f"Error initializing Pinecone index: {e}")

def generate_vector_id(source: str, chunk_index: int) -> str:
    """Generate a deterministic ID so re-runs upsert (not duplicate)."""
    raw = f"{source}::chunk_{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def get_base64_image(image_path):
    # Open the image from the path
    img = Image.open(image_path)
    
    # Save image to a BytesIO object
    buffered = BytesIO()
    # Use the image's original format (e.g., JPEG, PNG), defaulting to PNG if None
    img_format = img.format if img.format else 'PNG'
    img.save(buffered, format=img_format)
    
    # Encode the bytes to a base64 string
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def store_in_pinecone(records: list[dict], image_path_list: List[dict]) -> None:
    """Embed chunks and upsert them into Pinecone with full metadata."""
    index = init_pinecone_index()

    # Embed all enriched texts
    enriched_texts = [r["enriched_text"] for r in records]
    logging.info(f"Embedding {len(enriched_texts)} chunks with {EMBEDDING_MODEL}...")

    all_embeddings = []
    # Google AI Studio API supports batching
    try:
        for i in range(0, len(enriched_texts), BATCH_SIZE):
            batch = enriched_texts[i : i + BATCH_SIZE]
            batch_embeddings = get_document_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
    except Exception as e:
        logging.error(f"Error during embedding: {e}\nCheck your API key, model name, and network connection.")
        return

    # Build Pinecone vectors
    vectors = []
    for record, embedding in zip(records, all_embeddings):
        vector_id = generate_vector_id(
            record["metadata"]["source_document"],
            record["metadata"]["chunk_index"],
        )
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                **record["metadata"],
                "text": record["text"], # store raw text for retrieval
                "enriched_text": record["enriched_text"], # store enriched text too
            },
        })

    # Upsert in batches
    logging.info(f"Upserting {len(vectors)} vectors to Pinecone...")
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        index.upsert(vectors=batch, namespace='text&tables')
    
    #Upserting Image embeddings in 'images' namespace
    if image_path_list:
        for image_item in image_path_list:
            image_index = image_item["item_idx"]
            image_path = image_item["img_path"]
            image_embedding = embed_image(image_path)

            vector_id = generate_vector_id(
                image_path,
                image_index,
            )
            index.upsert(vectors=[{
                "id": vector_id,
                "values": image_embedding,
                "metadata": {
                    "source_document": image_path,
                    "base64_string": get_base64_image(image_path)
                }
            }], namespace='images')
    logging.info(f" Successfully stored {len(vectors)} chunks in '{PINECONE_INDEX_NAME}'")
    logging.info(f" Successfully stored {len(image_path_list)} images in '{PINECONE_INDEX_NAME}' under 'images' namespace")
    stats = index.describe_index_stats()
    logging.info(f"Index stats: {stats}")

def query_pinecone(query: str, top_k: int = 5, namespace: Optional[str] = None) -> list[dict]:
    """Query the index and return matching chunks with metadata."""
    query_embedding = get_query_embedding(query)

    index = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace='text&tables',
    )

    text_matches = []
    for match in results["matches"]:
        text_matches.append({
            "score": match["score"],
            "text": match["metadata"].get("enriched_text", ""),
            "source": match["metadata"].get("source_document", ""),
            "pages": match["metadata"].get("pages", []),
            "headings": match["metadata"].get("heading_trail", ""),
            "item_types": match["metadata"].get("item_types", []),
        })
    
    clip_vector = embed_image_query(query)
    image_results = index.query(
        vector=clip_vector,
        top_k=5,
        namespace='images',
        include_metadata=True
    )
    image_matches = []
    for match in image_results["matches"]:
        image_matches.append({
            "score": match["score"],
            "image_path": match["metadata"].get("source_document", ""),
            "base64_string": match["metadata"].get("base64_string", ""),
        })
    return text_matches, image_matches
