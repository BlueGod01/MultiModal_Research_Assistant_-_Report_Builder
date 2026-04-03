from fastapi import FastAPI, UploadFile, File
import aiofiles
import os
from Data_Ingestion_Pipeline.Data_Pipeline import run_parallel_pipeline

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()  # async I/O
        await f.write(content)
    return {"message": "Processing started", "file": file.filename}

@app.post("/Knowledge_base")
async def create_knowledge_base(UPLOAD_DIR = "uploads", parsing_strategy):
    run_parallel_pipeline(UPLOAD_DIR, max_workers=4, strategy=parsing_strategy)
    return {"message": "Knowledge base creation started"}
