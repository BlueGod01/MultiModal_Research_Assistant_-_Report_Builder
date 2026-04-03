import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from parser import convert_document
from chunker import extract_and_chunk
from Pinecone_Functions import store_in_pinecone

PDF_DIR = "uploads"
def pipeline(document_path: str, parsing_strategy: str = "medium"):
    try:
        # Step 1: Convert PDF to structured document
        print(f"Converting document '{document_path}'...")
        docs = convert_document(document_path, parsing_strategy=parsing_strategy)

        # Step 2: Extract content and chunk
        for doc in docs:
            record, image_map = extract_and_chunk(doc, document_path=document_path)
            print(f"Extracted {len(record)} records from {os.path.basename(document_path)}.")
            
            # Step 3: Store chunks in Pinecone
            store_in_pinecone(records=record, image_path_list=image_map)
            
    except Exception as e:
        print(f"Error processing {document_path}: {e}")

def run_parallel_pipeline(directory_path: str, max_workers: int = None, strategy: str = "medium")-> None:
    """
    Extracts all PDFs from a directory and processes them in parallel.
    - max_workers: Limits parallel processes to prevent memory/CPU exhaustion. 
      Defaults to number of processors if None.
    """
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    # Gather all PDF paths from directory
    pdf_paths = [
        os.path.join(directory_path, f) 
        for f in os.listdir(directory_path) 
        if f.lower().endswith('.pdf')
    ]

    if not pdf_paths:
        print("No PDF files found in the directory.")
        return

    # ProcessPoolExecutor for parallelism (bypassing GIL)
    # Recommended: Set max_workers to a value that balances CPU cores and RAM
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function to fix the parsing_strategy argument
        worker_func = partial(pipeline, parsing_strategy=strategy)
        
        # Map the pipeline function to the list of paths
        future_to_pdf = {executor.submit(worker_func, path): path for path in pdf_paths}
        
        for future in as_completed(future_to_pdf):
            pdf = future_to_pdf[future]
            try:
                future.result()
            except Exception as exc:
                print(f"File {pdf} generated an unhandled exception: {exc}")

