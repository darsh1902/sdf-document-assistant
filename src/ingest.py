"""
PDF ingestion: turn a PDF into a list of pages with clean text.

Digital pages are read directly with PyMuPDF. Scanned pages have (almost) no text layer,
so for those we render the page as an image and run Tesseract OCR on it.
"""

import re
import sys
from pathlib import Path

import pymupdf  # PyMuPDF (older code imports this as `fitz`)
import pytesseract
from PIL import Image

# Works both when imported as `src.ingest` (from app.py) and when run directly
# as `python src/ingest.py` (where src/ itself is on the import path).
try:
    from src.config import OCR_DPI, OCR_MIN_CHARS, SAMPLE_PDF
except ImportError:
    from config import OCR_DPI, OCR_MIN_CHARS, SAMPLE_PDF


def clean_text(text: str) -> str:
    """Light cleanup so chunking and embeddings aren't thrown off by junk.

    We keep line breaks (and blank lines between paragraphs) because chunking.py
    uses them to split on natural boundaries.
    """
    # Remove control characters and the "unknown character" symbol (�) that
    # PDFs/OCR sometimes produce. \n is kept on purpose.
    text = re.sub(r"[\x00-\x09\x0b-\x1f\x7f�]", " ", text)
    # Collapse runs of spaces/tabs inside a line into a single space.
    text = re.sub(r"[ \t ]+", " ", text)
    # Trim spaces at the start/end of every line.
    text = "\n".join(line.strip() for line in text.splitlines())
    # Allow at most one blank line in a row (= a paragraph break).
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ocr_page(page: pymupdf.Page) -> str:
    """Render one PDF page as a grayscale image and read its text with Tesseract."""
    # Rendering straight to grayscale is smaller than RGB and OCRs just as well.
    pix = page.get_pixmap(dpi=OCR_DPI, colorspace=pymupdf.csGRAY)
    image = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(image)


def load_pdf(path: str | Path) -> list[dict]:
    """Read a PDF and return one dict per page.

    Each dict looks like:
        {"page_num": 1, "text": "...", "ocr_used": False, "source": "file.pdf"}
    """
    path = Path(path)
    source = path.name
    pages = []

    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc):
            page_num = index + 1  # humans (and citations) count pages from 1
            text = clean_text(page.get_text())
            ocr_used = False

            # Very little text usually means the page is a scanned image.
            if len(text) < OCR_MIN_CHARS:
                try:
                    ocr_text = clean_text(ocr_page(page))
                except pytesseract.TesseractNotFoundError:
                    print(f"Page {page_num}: needs OCR, but Tesseract isn't installed "
                          "(macOS: `brew install tesseract`). Keeping the little text we have.")
                    ocr_text = ""

                # Only switch to the OCR result if it actually found more text.
                if len(ocr_text) > len(text):
                    text = ocr_text
                    ocr_used = True
                    print(f"Page {page_num}: OCR used ({len(text)} chars)")
                else:
                    print(f"Page {page_num}: OCR found no extra text (page may be blank)")
            else:
                print(f"Page {page_num}: digital text ({len(text)} chars)")

            pages.append({
                "page_num": page_num,
                "text": text,
                "ocr_used": ocr_used,
                "source": source,
            })

    return pages


if __name__ == "__main__":
    # Quick test: `python src/ingest.py` uses the sample PDF,
    # or pass another one: `python src/ingest.py data/other.pdf`
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE_PDF
    print(f"Loading {pdf_path} ...\n")

    pages = load_pdf(pdf_path)

    print(f"\n=== {len(pages)} pages from {pdf_path.name} ===")
    for p in pages:
        method = "OCR" if p["ocr_used"] else "digital"
        preview = p["text"][:200].replace("\n", " | ")
        print(f"\n--- Page {p['page_num']} ({method}, {len(p['text'])} chars) ---")
        print(preview + ("..." if len(p["text"]) > 200 else ""))

    ocr_count = sum(p["ocr_used"] for p in pages)
    print(f"\nSummary: {len(pages)} pages, {ocr_count} needed OCR, "
          f"{len(pages) - ocr_count} had digital text.")
