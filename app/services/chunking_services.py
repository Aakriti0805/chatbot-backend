from llama_index.core.node_parser import SentenceSplitter  # type: ignore
import re


splitter = SentenceSplitter(chunk_size=300, chunk_overlap=65)

MIN_CHUNK_LENGTH = 30


def is_low_value_chunk(text: str) -> bool:
    """
    Returns True for chunks that are almost certainly not useful —
    blank pages, bare chapter/section headers, or near-empty fragments.
    """
    stripped = text.strip()

    if len(stripped) < MIN_CHUNK_LENGTH:
        return True

    if "INTENTIONALLY LEFT BLANK" in stripped.upper():
        return True

    has_sentence_punctuation = bool(re.search(r'[.!?]\s+[A-Z]', stripped))
    word_count = len(stripped.split())

    if word_count <= 12 and not has_sentence_punctuation:
        return True

    return False


def _split_sections(text: str):
    """
    Splits a page's text into sections based on the "[TABLES]" and "[IMAGES]"
    markers that textextraction_service.py inserts (main text, then
    "\\n\\n[TABLES]\\n" + Camelot table blocks, then "\\n\\n[IMAGES]\\n" + OCR blocks).

    This separation matters because SentenceSplitter splits on sentence
    punctuation — it doesn't understand pipe-separated table rows, and can
    split a table right down the middle. Tables/images are kept separate so
    they can be treated as atomic chunks (as-is, not split further).
    """
    main_text = text
    table_blocks: list[str] = []
    image_blocks: list[str] = []

    if "\n\n[TABLES]\n" in main_text:
        main_text, _, rest = main_text.partition("\n\n[TABLES]\n")
        if "\n\n[IMAGES]\n" in rest:
            tables_part, _, images_part = rest.partition("\n\n[IMAGES]\n")
            table_blocks = [b.strip() for b in tables_part.split("\n\n") if b.strip()]
            image_blocks = [b.strip() for b in images_part.split("\n\n") if b.strip()]
        else:
            table_blocks = [b.strip() for b in rest.split("\n\n") if b.strip()]
    elif "\n\n[IMAGES]\n" in main_text:
        main_text, _, images_part = main_text.partition("\n\n[IMAGES]\n")
        image_blocks = [b.strip() for b in images_part.split("\n\n") if b.strip()]

    return main_text, table_blocks, image_blocks


def chunk_text(pages_text: list[dict]) -> list[dict]:
    all_chunks = []

    for page in pages_text:
        text = page["text"]
        page_number = page["page_number"]

        if not text.strip():
            continue

        main_text, table_blocks, image_blocks = _split_sections(text)

        if main_text.strip():
            chunks = splitter.split_text(main_text)
            for chunk in chunks:
                cleaned = chunk.strip()
                if is_low_value_chunk(cleaned):
                    continue
                all_chunks.append({"text": cleaned, "page_number": page_number})


        for block in table_blocks:
            if len(block) >= MIN_CHUNK_LENGTH:
                all_chunks.append({"text": block, "page_number": page_number})

        
        for block in image_blocks:
            if not is_low_value_chunk(block):
                all_chunks.append({"text": block, "page_number": page_number})

    return all_chunks