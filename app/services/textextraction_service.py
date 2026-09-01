import fitz  # PyMuPDF #type: ignore
import camelot  #type: ignore
import numpy as np
from paddleocr import PaddleOCR  # type: ignore
from pdf2image import convert_from_path  # type: ignore
from PIL import Image
import io

ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", enable_mkldnn=False)


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


def run_ocr(image_np):
    """Common helper: run OCR + sort result, used everywhere"""
    result = ocr_engine.ocr(image_np)

    if result is None or len(result) == 0:
        return ""

    page_result = result[0]

    # kabhi kabhi PaddleOCR empty/no-detection pe raw numpy array
    # ya None return karta hai instead of dict-like OCRResult —
    # usko yahi filter kar do, bool() mat lagao directly
    if page_result is None or not hasattr(page_result, "get"):
        return ""

    rec_texts = page_result.get("rec_texts", [])
    rec_boxes = page_result.get("rec_boxes", [])

    if rec_boxes is not None and len(rec_boxes) > 0:
        return sort_ocr_by_position(rec_texts, rec_boxes)

    # FIX: rec_texts can be a NumPy array in newer PaddleOCR versions.
    # `if rec_texts:` raises "truth value of an array with more than
    # one element is ambiguous" on multi-element arrays — always check
    # length explicitly instead.
    if len(rec_texts) > 0:
        return "\n".join(rec_texts)

    return ""


def extract_images_ocr_from_page(doc, page, page_number, min_size=50):
    image_texts = []
    image_list = page.get_images(full=True)

    for img_index, img in enumerate(image_list):
        xref = img[0]
        try:
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            if pil_img.width < min_size or pil_img.height < min_size:
                continue

            image_np = np.array(pil_img)
            text = run_ocr(image_np)
            if text.strip():
                image_texts.append(text.strip())
        except Exception as e:
            print(f"Image OCR failed on page {page_number}, image {img_index}: {e}")

    return image_texts


def extract_text_from_pdf(filepath: str) -> list[dict]:
    doc = fitz.open(filepath)
    pages_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_number = page_num + 1
        text = page.get_text(sort=True).strip()

        if text:
            # Text-based page — combine with Camelot tables
            table_blocks = []
            try:
                tables = camelot.read_pdf(
                    filepath,
                    pages=str(page_number),
                    flavor="lattice"
                )
                for table in tables:
                    rows_text = []
                    for row in table.df.values.tolist():
                        cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                        if any(cleaned_row):
                            rows_text.append(" | ".join(cleaned_row))
                    if rows_text:
                        table_blocks.append("\n".join(rows_text))
            except Exception as e:
                print(f"Camelot table extraction failed on page {page_number}: {e}")

            combined_text = text
            if table_blocks:
                combined_text += "\n\n[TABLES]\n" + "\n\n".join(table_blocks)

            image_blocks = extract_images_ocr_from_page(doc, page, page_number)
            if image_blocks:
                combined_text += "\n\n[IMAGES]\n" + "\n\n".join(image_blocks)

        else:
            # Scanned page — no text at all, run OCR on full page image
            images = convert_from_path(
                filepath, first_page=page_number, last_page=page_number
            )
            image_np = np.array(images[0])
            combined_text = run_ocr(image_np).strip()

        pages_text.append({
            "page_number": page_number,
            "text": combined_text.strip()
        })

        print(f"Processed page {page_number}/{len(doc)}")

    doc.close()
    return pages_text


def extract_text_from_image(filepath: str) -> list[dict]:
    image = Image.open(filepath).convert("RGB")
    image_np = np.array(image)

    ocr_text = run_ocr(image_np)

    return [
        {
            "page_number": 1,
            "text": ocr_text.strip()
        }
    ]