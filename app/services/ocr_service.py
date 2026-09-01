import fitz  # PyMuPDF #type: ignore
import numpy as np
from paddleocr import PaddleOCR  # type: ignore
from pdf2image import convert_from_path  # type: ignore

ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", enable_mkldnn=False)


def extract_text_with_ocr(filepath: str) -> list[dict]:
    doc = fitz.open(filepath)
    pages_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()

        if text:
            # PyMuPDF ko text mil gaya — seedha use karo, OCR ki zarurat nahi
            pages_text.append({
                "page_number": page_num + 1,
                "text": text
            })
        else:
            # Text nahi mila — matlab scanned image hai, isi page ko OCR karo
            images = convert_from_path(
                filepath, first_page=page_num + 1, last_page=page_num + 1
            )
            image_np = np.array(images[0])
            result = ocr_engine.ocr(image_np)

            ocr_text = ""
            if result and result[0]:
                rec_texts = result[0].get("rec_texts", [])
                ocr_text = "\n".join(rec_texts)

            pages_text.append({
                "page_number": page_num + 1,
                "text": ocr_text.strip()
            })

        print(f"Processed page {page_num + 1}/{len(doc)}")

    doc.close()
    return pages_text