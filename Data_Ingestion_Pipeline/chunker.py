from docling.chunking import HybridChunker
from parser import save_images
#this is Docling's recommended chunker for RAG. 
# It respects document structure, splits oversized tables with repeated headers, and merges small sibling chunks

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MAX_TOKENS_PER_CHUNK = 512
def extract_and_chunk(doc, document_path: str = None) -> list[dict]:
    """
    Extract all content from a PDF and return chunked records
    suitable for vector database storage.
    Each record contains:
      - text: the chunk text
      - enriched_text: text with section headings prepended (use this for embeddings)
      - metadata: page numbers, bounding boxes, item types, headings
    """
    

    # --- Save images to disk ---
    image_map = save_images(doc)

    # --- Chunk the document ---
    chunker = HybridChunker(
        max_tokens=MAX_TOKENS_PER_CHUNK,
        merge_peers=True,
        repeat_table_header=True, # each table chunk is self-contained
    )

    chunks = chunker.chunk(dl_doc=doc)
    records = []

    for i, chunk in enumerate(chunks):
        # Contextualized text prepends section headings — best for embeddings/RAG
        enriched_text = chunker.contextualize(chunk = chunk)

        # Collect metadata from each doc item in the chunk
        pages = set()
        bboxes = []
        item_types = set()

        for doc_item in chunk.meta.doc_items:
            # Track the type of content (text, table, picture, formula, etc.)
            if hasattr(doc_item, "label"):
                item_types.add(str(doc_item.label))

            # Extract provenance (page number + bounding box)
            for prov in getattr(doc_item, "prov", []):
                pages.add(prov.page_no)
                bboxes.append({
                    "page": prov.page_no,
                    "bbox": {
                        "l": prov.bbox.l,
                        "t": prov.bbox.t,
                        "r": prov.bbox.r,
                        "b": prov.bbox.b,
                    },
                })

        # Gather heading trail from chunk metadata
        headings = []
        for heading in chunk.meta.headings:
            headings.append(heading)

        record = {
            "chunk_id": i,
            "text": chunk.text,
            "enriched_text": enriched_text,
            "metadata": {
                "source": document_path,
                "pages": sorted(pages),
                "headings": headings,
                "item_types": sorted(item_types),
                "bounding_boxes": bboxes,
            },
        }
        records.append(record)       
    return records, image_map