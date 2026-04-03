import fitz  # PyMuPDF
import asyncio
from typing import Dict, List
from PIL import Image
from io import BytesIO
import base64

# Docling
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.settings import settings


import os
import json
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")# Prevents thread oversubscription on CPU-bound work

def convert_large_document(document_path: str, parsing_strategy: str = "medium"):
    """
    Convert a PDF with parallel page processing.
    Docling's StandardPdfPipeline internally uses multithreaded
    stage-based parallelism (OCR, layout, table extraction run
    concurrently on different pages).
    """
    if parsing_strategy == "fast":
        # Fast PyMuPDF backend with minimal processing
        pipeline_options = PdfPipelineOptions(
            pdf_backend="pymupdf",
            do_ocr=False,
            do_table_structure=False,
            do_formula_enrichment=False,
                # ── Parallelism tuning ──
        # Increase batch sizes so more pages are processed per stage iteration
        ocr_batch_size=8, # default 4 — pages batched for OCR
        layout_batch_size=8, # default 4 — pages batched for layout detection
        table_batch_size=8, # default 4 — pages batched for table extraction
        )
        
    elif parsing_strategy == "medium":
        # pypdfium2 backend with table structure and formula enrichment
        pipeline_options = PdfPipelineOptions(
            pdf_backend="pypdfium2",
            do_ocr=False,
            do_table_structure=True,
            do_formula_enrichment=True,
            table_mode=TableFormerMode.ACCURATE,
            # ── Parallelism tuning ──
        # Increase batch sizes so more pages are processed per stage iteration
        ocr_batch_size=8, # default 4 — pages batched for OCR
        layout_batch_size=8, # default 4 — pages batched for layout detection
        table_batch_size=8, # default 4 — pages batched for table extraction
        )

    elif parsing_strategy == "deep":
        # Full OCR with table structure and formula enrichment
        pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=EasyOcrOptions(lang=["en"], use_gpu=True),
        do_table_structure=True,
        do_formula_enrichment=True,
        table_mode=TableFormerMode.ACCURATE,
        generate_picture_images=True,
        images_scale=2.0,

        # ── Parallelism tuning ──
        # Increase batch sizes so more pages are processed per stage iteration
        ocr_batch_size=8, # default 4 — pages batched for OCR
        layout_batch_size=8, # default 4 — pages batched for layout detection
        table_batch_size=8, # default 4 — pages batched for table extraction
    )
    pipeline_options.page_chunk_size = 50

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, vlm_options=(VlmPipeline.VlmPipelineOptions(enable_vlm=True, vlm_model="google/flan-t5-small")if parsing_strategy == "deep" else None)  )
        }
    )

    # Must use convert_all() with page_chunk_size
    results = list(converter.convert_all([document_path]))
    return [res.document for res in results]

def convert_document(document_path: str, parsing_strategy: str = "medium"):
    """Convert a PDF with all extraction features enabled."""
    doc = fitz.open(document_path)
    if doc.page_count == 0:
        raise ValueError("Document has no pages")
    elif doc.page_count > 100:
        return convert_large_document(document_path, parsing_strategy)
    if parsing_strategy == "fast":
        # Fast PyMuPDF backend with minimal processing
        pipeline_options = PdfPipelineOptions(
            pdf_backend="pymupdf",
            do_ocr=False,
            do_table_structure=False,
            do_formula_enrichment=False,
        )
    elif parsing_strategy == "medium":
        # pypdfium2 backend with table structure and formula enrichment
        pipeline_options = PdfPipelineOptions(
            pdf_backend="pypdfium2",
            do_ocr=False,
            do_table_structure=True,
            do_formula_enrichment=True,
            table_mode=TableFormerMode.ACCURATE,
        )

    elif parsing_strategy == "deep":
        # Full OCR with table structure and formula enrichment
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=EasyOcrOptions(lang=["en"], use_gpu=True),
            do_table_structure=True,
            do_formula_enrichment=True,
            table_mode=TableFormerMode.ACCURATE,
            images_scale = 1.0
        )
    doc.close()
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, vlm_options=(VlmPipeline.VlmPipelineOptions(enable_vlm=True, vlm_model="google/flan-t5-small")if parsing_strategy == "deep" else None))
        }
    )

    return [converter.convert(document_path).document]

def save_images(doc, output_dir: str = "extracted_images") -> List[Dict]:
    """Save extracted picture items as image files. Returns a map of item ref -> file path."""
    os.makedirs(output_dir, exist_ok=True)
    image_map = []

    for item_idx, (item, _level) in enumerate(doc.iterate_items()):
        if hasattr(item, "image") and item.image is not None:
            img_path = os.path.join(output_dir, f"image_{item_idx}.png")
            item.image.pil_image.save(img_path)
            image_map.append({"item_idx": item_idx, "img_path": img_path})
    return image_map 