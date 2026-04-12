from Data_Ingestion_Pipeline.Pinecone_Functions import query_pinecone
from typing import List, Dict
from Brain import google_llm
llm = google_llm(model="gemini-1.5-flash")
def is_low_quality(docs, threshold=0.3):
    # assuming Pinecone score in metadata
    scores = [d.get("score", 0) for d in docs]
    return len(scores) == 0 or max(scores) < threshold
def retrieve_multimodal(query: str, top_k: int) -> List[Dict]:
    text,images =  query_pinecone(query, top_k)
    if not is_low_quality(text, threshold=0.3):
        text_content = []
        for i, doc in enumerate(text):
            text_content.append(f"Text: {doc['text']}\nPage_no: {doc['pages']}\nSource: {doc['source']}\nHeading_trail: {doc['headings']}\n")
    else:
        text_content.append("No relevant text found.")
    if not is_low_quality(images, threshold=0.3):
        image_content = []
        for i, img in enumerate(images):
            image_content.append({
                "image": img['base64_string'],
                "image_path": img['image_path']
            })
    else:
        image_content.append("No relevant images found.")
    return text_content, image_content
def refine_query(query, attempt):
    prompt = f"""
    Improve the query for better document retrieval.

    Original Query: {query}
    Attempt: {attempt}

    Return a more specific and information-rich query, make sure to include all important keywords and context.
    Keep it small and simple
    If the query is already information-rich and matches with the context then dont change it.
    Dont put extra, unnecessary informations.
    """
    return llm.invoke(prompt).content
def query_corrected_retrieval(query, max_retries=2):
    for attempt in range(max_retries + 1):
        query = refine_query(query, attempt)

    docs,images = retrieve_multimodal(query,top_k=20)

    return docs,images
