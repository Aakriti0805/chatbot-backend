from llama_index.core.node_parser import SentenceSplitter  # type: ignore
import re

splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)

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


def chunk_text(pages_text: list[dict]) -> list[dict]:
    all_chunks = []

    for page in pages_text:
        text = page["text"]
        page_number = page["page_number"]

        if not text.strip():
            continue

        chunks = splitter.split_text(text)

        for chunk in chunks:
            cleaned = chunk.strip()

            if is_low_value_chunk(cleaned):
                continue

            all_chunks.append({
                "text": cleaned,
                "page_number": page_number
            })

    return all_chunks