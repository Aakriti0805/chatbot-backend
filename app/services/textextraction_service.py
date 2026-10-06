import fitz  # PyMuPDF #type: ignore
import camelot  #type: ignore
import numpy as np
from paddleocr import PaddleOCR  # type: ignore
from pdf2image import convert_from_path  # type: ignore
from PIL import Image
import gc  # NEW: Camelot ke objects hatane ke baad handle chhodne ke liye
import io
import logging
import os
import shutil  # NEW: Camelot ke liye temp copy banane ke liye
import tempfile  # NEW: temp folder banane ke liye
import threading
import time
import warnings

logger = logging.getLogger(__name__)

# OCR speed settings (all can be overridden from .env).
# - OCR_DPI: resolution used when a scanned PDF page is rendered to an image.
#   150 is enough for typed letters/forms; 200+ makes OCR much slower.
# - OCR_MAX_SIDE: the longest side of any image is shrunk to this many pixels
#   before OCR. Text detection time grows with the number of pixels.
# Camelot warns once per scanned page ("page-1 is image-based"). Scanned pages
# are handled by OCR, so this warning is just noise.
warnings.filterwarnings("ignore", message=".*is image-based.*", category=UserWarning)

OCR_DPI = int(os.getenv("OCR_DPI", "150"))
OCR_MAX_SIDE = int(os.getenv("OCR_MAX_SIDE", "1600"))


def _build_ocr_engine():
    """
    Builds the PaddleOCR engine. Document unwarping and document/text-line
    orientation checks are switched off: they are extra neural networks that
    run on every page, are very slow on CPU, and are not needed for upright,
    scanned railway letters and manuals. If the installed PaddleOCR version
    does not accept these options, it falls back to the original settings.
    """
    try:
        return PaddleOCR(
            lang="en",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except Exception as exc:
        logger.warning("Fast PaddleOCR options not supported, using defaults: %s", exc)
        return PaddleOCR(use_angle_cls=True, lang="en", enable_mkldnn=False)


ocr_engine = _build_ocr_engine()

# One PaddleOCR instance is shared by all requests. FastAPI runs sync
# endpoints in a thread pool, so two uploads at once could call the engine
# concurrently, which is not safe. Serialize the calls.
_ocr_lock = threading.Lock()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DOCX_EXTENSIONS = {".docx"}


def sort_ocr_by_position(rec_texts, rec_boxes, y_tolerance=15):
    """rec_boxes format: [x1, y1, x2, y2] per detected text"""
    items = [
        {"text": t, "x": box[0], "y": box[1]}
        for t, box in zip(rec_texts, rec_boxes)
    ]
    items.sort(key=lambda i: i["y"])

    rows = []
    current_row = []
    current_y = None
    for item in items:
        if current_y is None or abs(item["y"] - current_y) <= y_tolerance:
            current_row.append(item)
            current_y = item["y"] if current_y is None else current_y
        else:
            rows.append(current_row)
            current_row = [item]
            current_y = item["y"]
    if current_row:
        rows.append(current_row)

    lines = []
    for row in rows:
        row.sort(key=lambda i: i["x"])
        lines.append(" ".join(i["text"] for i in row))
    return "\n".join(lines)


def _shrink_for_ocr(image_np):
    """Scales the image down so its longest side is at most OCR_MAX_SIDE pixels."""
    height, width = image_np.shape[:2]
    longest = max(height, width)
    if longest <= OCR_MAX_SIDE:
        return image_np

    scale = OCR_MAX_SIDE / longest
    resized = Image.fromarray(image_np).resize(
        (int(width * scale), int(height * scale)), Image.LANCZOS
    )
    return np.array(resized)


def run_ocr(image_np):
    """Common helper: run OCR + sort result, used everywhere"""
    image_np = _shrink_for_ocr(image_np)

    started = time.perf_counter()
    with _ocr_lock:
        result = ocr_engine.ocr(image_np)
    logger.debug(
        "OCR on %dx%d image took %.1fs",
        image_np.shape[1], image_np.shape[0], time.perf_counter() - started,
    )

    if result is None or len(result) == 0:
        return ""

    page_result = result[0]

    
    if page_result is None or not hasattr(page_result, "get"):
        return ""

    rec_texts = page_result.get("rec_texts", [])
    rec_boxes = page_result.get("rec_boxes", [])

    if rec_boxes is not None and len(rec_boxes) > 0:
        return sort_ocr_by_position(rec_texts, rec_boxes)

    
    if len(rec_texts) > 0:
        return "\n".join(rec_texts)

    return ""


def extract_images_ocr_from_page(doc, page, page_number, ocr_cache, min_size=100):
    """
    OCRs the images embedded in one PDF page.

    `ocr_cache` is shared across the whole document: an image (xref) is
    OCR'd only once. Logos, headers and watermarks repeat on every page, and
    OCR on CPU is the slowest step, so repeating it per page made large
    uploads look like they hang. Images smaller than `min_size` pixels
    (icons, bullets, lines) are skipped too.
    """
    image_texts = []

    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        if xref in ocr_cache:
            continue
        ocr_cache[xref] = ""  # mark as seen, even if it turns out unusable

        try:
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            if pil_img.width < min_size or pil_img.height < min_size:
                continue

            text = run_ocr(np.array(pil_img)).strip()
            ocr_cache[xref] = text
            if text:
                image_texts.append(text)
        except Exception as e:
            logger.warning("Image OCR failed on page %s, image %s: %s", page_number, img_index, e)

    return image_texts


def extract_tables_by_page(filepath: str) -> dict[int, list[str]]:
    """
    Runs Camelot ONCE for the whole PDF and returns {page_number: [table_text, ...]}.

    Calling camelot.read_pdf() separately for every page re-reads and
    re-processes the entire file each time, which is very slow on
    multi-page PDFs. If Camelot fails (e.g. Ghostscript missing), the PDF is
    still processed, just without tables.
    """
    tables_by_page: dict[int, list[str]] = {}
    started = time.perf_counter()

    # CHANGED: Camelot PDF ka file handle band nahi karta, jisse Windows pe original
    # file lock ho jati hai (delete pe PermissionError). Isliye Camelot ko ek temp COPY
    # pe chalate hain aur copy baad mein delete kar dete hain.
    temp_dir = tempfile.mkdtemp(prefix="camelot_")
    temp_pdf = os.path.join(temp_dir, "input.pdf")
    tables = None

    try:
        shutil.copyfile(filepath, temp_pdf)
        tables = camelot.read_pdf(temp_pdf, pages="all", flavor="lattice")

        for table in tables:
            rows_text = []
            for row in table.df.values.tolist():
                cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                if any(cleaned_row):
                    rows_text.append(" | ".join(cleaned_row))
            if rows_text:
                tables_by_page.setdefault(int(table.page), []).append("\n".join(rows_text))
    except Exception as e:
        logger.warning("Camelot table extraction failed, continuing without tables: %s", e)
        return tables_by_page
    finally:
        # Camelot ke objects hatao taaki handle chhoot jaye, phir temp folder saaf karo
        del tables
        gc.collect()
        shutil.rmtree(temp_dir, ignore_errors=True)

    logger.debug(
        "Camelot found tables on %d page(s) in %.1fs",
        len(tables_by_page), time.perf_counter() - started,
    )
    return tables_by_page


def extract_text_from_pdf(filepath: str) -> list[dict]:
    started = time.perf_counter()
    # CHANGED: doc ko pehle None rakha aur fitz.open ko try ke andar daala, taaki
    # open ke baad beech mein koi bhi error aaye to bhi doc.close() zaroor chale.
    doc = None
    pages_text = []

    try:
        doc = fitz.open(filepath)
        total_pages = len(doc)

        logger.debug("PDF extraction started: %s (%d pages)", os.path.basename(filepath), total_pages)

        tables_by_page = extract_tables_by_page(filepath)
        ocr_cache: dict[int, str] = {}

        for page_num in range(total_pages):
            page = doc[page_num]
            page_number = page_num + 1
            text = page.get_text(sort=True).strip()

            if text:
                # Text-based page: add its tables and any text found in images.
                combined_text = text

                table_blocks = tables_by_page.get(page_number, [])
                if table_blocks:
                    combined_text += "\n\n[TABLES]\n" + "\n\n".join(table_blocks)

                image_blocks = extract_images_ocr_from_page(doc, page, page_number, ocr_cache)
                if image_blocks:
                    combined_text += "\n\n[IMAGES]\n" + "\n\n".join(image_blocks)

            else:
                # Scanned page: no text layer at all, run OCR on the full page image.
                logger.debug("Page %d has no text layer, running OCR", page_number)
                images = convert_from_path(
                    filepath,
                    first_page=page_number,
                    last_page=page_number,
                    dpi=OCR_DPI,
                )
                combined_text = run_ocr(np.array(images[0])).strip()

            pages_text.append({
                "page_number": page_number,
                "text": combined_text.strip(),
            })

            logger.info("Processed page %d/%d", page_number, total_pages)
    finally:
        if doc is not None:
            doc.close()

    logger.debug("PDF extraction finished in %.1fs", time.perf_counter() - started)
    return pages_text


def extract_text_from_image(filepath: str) -> list[dict]:
    # CHANGED: with use kiya hai taaki image ki file turant band ho jaye
    # (Windows pe delete ke liye zaroori hai)
    with Image.open(filepath) as img:
        image_np = np.array(img.convert("RGB"))

    ocr_text = run_ocr(image_np)

    return [
        {
            "page_number": 1,
            "text": ocr_text.strip()
        }
    ]


def extract_text_from_plain_text(text: str) -> list[dict]:
    """Direct text input (no file uploaded) — no extraction needed."""
    return [{"page_number": 1, "text": text.strip()}]


def extract_text_from_docx(filepath: str) -> list[dict]:
    """
    Extracts paragraphs and tables from a .docx file.

    A .docx file has no fixed page layout, so the whole document is returned
    as a single "page" (page_number=1). Tables are appended after the
    "[TABLES]" marker so chunking_services.py keeps each one as its own chunk.
    """
    try:
        import docx  # type: ignore  # provided by the python-docx package
    except ImportError as exc:
        raise RuntimeError(
            "DOCX support needs the python-docx package: pip install python-docx"
        ) from exc

    document = docx.Document(filepath)

    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]

    table_blocks = []
    for table in document.tables:
        rows_text = []
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if any(cells):
                rows_text.append(" | ".join(cells))
        if rows_text:
            table_blocks.append("\n".join(rows_text))

    text = "\n".join(paragraphs)
    if table_blocks:
        text += "\n\n[TABLES]\n" + "\n\n".join(table_blocks)

    # rstrip only: a leading "\n\n[TABLES]" must survive for the chunker.
    return [{"page_number": 1, "text": text.rstrip()}]


def process_file(filepath: str) -> list[dict]:
    """Entry point — routes to the right extractor based on file extension."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext in IMAGE_EXTENSIONS:
        return extract_text_from_image(filepath)
    elif ext in DOCX_EXTENSIONS:
        return extract_text_from_docx(filepath)
    else:
        raise ValueError(f"Unsupported file type for ingestion: {ext}")