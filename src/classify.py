"""
Page classification: guess which kind of document each page belongs to, then group
consecutive pages of the same kind into "documents" (with page ranges).

Two layers, so it works on SDF packets from any supplier:
1. Keyword scoring (fast, free, explainable). Words in the page's title area count
   double, because a title like "QUALITY CERTIFICATE" says more than a word buried in
   the body (a cover letter that *lists* the attached documents mentions all of them).
2. LLM fallback: if the keywords aren't confident, the LLM reads the top of the page
   and picks a label. That handles wording our keyword lists have never seen.
"""

import re
import sys
from pathlib import Path

try:
    from src.config import CLASSIFIER_HEADER_CHARS, SAMPLE_PDF, USE_LLM_CLASSIFIER
    from src.ingest import load_pdf
    from src.llm import ERROR_PREFIX, call_llm
except ImportError:
    from config import CLASSIFIER_HEADER_CHARS, SAMPLE_PDF, USE_LLM_CLASSIFIER
    from ingest import load_pdf
    from llm import ERROR_PREFIX, call_llm

# ---------------------------------------------------------------------------
# Keyword rules — edit here to change classification.
# "strong" = phrases that almost only appear in that document type.
# "weak"   = words that hint at the type but also show up elsewhere.
# Each keyword counts once per page, so a page that repeats "specification"
# 20 times doesn't outscore a page with a real title.
# Write keywords in lowercase, with spaces instead of hyphens.
# ---------------------------------------------------------------------------
DOC_TYPE_RULES = {
    "Cover Letter": {
        "strong": ["dear", "sincerely", "to whom it may concern", "cover letter",
                   "best regards", "kind regards", "please find attached", "forwarded message"],
        "weak": ["regards", "subject", "attached"],
    },
    "Certificate of Quality": {
        "strong": ["certificate of quality", "quality certificate", "certificate of analysis",
                   "certificate of conformance", "certificate of conformity"],
        "weak": ["lot", "batch", "specification", "result", "expiration", "expiry", "retest",
                 "release"],
    },
    "BSE/TSE Declaration": {
        "strong": ["animal origin", "transmissible spongiform", "bse/tse", "tse/bse",
                   "animal derived"],
        # "bse"/"tse" are weak because other pages mention them in passing.
        "weak": ["bse", "tse", "bovine"],
    },
    "Packaging Specification": {
        "strong": ["packaging specification", "packaging component", "packaging configuration"],
        "weak": ["packaging", "container", "label", "carton", "pallet"],
    },
    "Material Specification": {
        "strong": ["material description", "materials of construction", "material specification",
                   "bill of materials", "material composition"],
        "weak": ["polymer", "elastomer", "usp class vi"],
    },
    "Sterilization Certificate": {
        "strong": ["certificate of processing", "sterilization certificate",
                   "certificate of irradiation", "dose audit"],
        "weak": ["sterilization", "gamma", "irradiation", "kgy", "dosimeter"],
    },
    "Supplier Qualification": {
        "strong": ["supplier qualification", "vendor qualification", "qualification record",
                   "supplier questionnaire"],
        "weak": ["audit", "qualification", "approved supplier"],
    },
    "Compliance Statement": {
        "strong": ["compliance bulletin", "compliance statement", "regulatory statement",
                   "statement of compliance", "regulatory compliance"],
        "weak": ["reach", "rohs", "nitrosamine", "elemental impurities", "residual solvents"],
    },
}
HEADER_POINTS = 6   # strong keyword in the page's title area
STRONG_POINTS = 3   # strong keyword anywhere else on the page
WEAK_POINTS = 1

# A page needs at least this score to get a type; otherwise it's "Other".
# 3 = one strong keyword, or three different weak ones.
MIN_SCORE = 3
# If the runner-up is this close to the winner, the keywords aren't sure -> ask the LLM.
CLOSE_CALL_MARGIN = 1

OTHER = "Other"


def _normalize(text: str) -> str:
    """Lowercase, hyphens -> spaces, and all whitespace -> one space.

    Collapsing newlines matters: PDFs often break a title across lines
    ("Certificate of\\nQuality"), which would otherwise never match a phrase.
    """
    return re.sub(r"\s+", " ", text.lower().replace("-", " "))


def _contains(text: str, keyword: str) -> bool:
    """Whole-word match, so "lot" doesn't match "pilot" and "result" doesn't match "Results"."""
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def score_text(text: str, header_chars: int = 0) -> dict[str, int]:
    """Return a score for every document type, e.g. {"Cover Letter": 3, ...}.

    `header_chars` > 0 treats the first N characters as the page's title area, where
    strong keywords earn HEADER_POINTS. Leave it at 0 for short texts like questions.
    """
    text = _normalize(text)
    header = text[:header_chars]
    scores = {}
    for doc_type, rules in DOC_TYPE_RULES.items():
        score = 0
        for kw in rules["strong"]:
            if header and _contains(header, kw):
                score += HEADER_POINTS
            elif _contains(text, kw):
                score += STRONG_POINTS
        score += WEAK_POINTS * sum(_contains(text, kw) for kw in rules["weak"])
        scores[doc_type] = score
    return scores


def classify_page(text: str, min_score: int = MIN_SCORE, header_chars: int = 0) -> str:
    """Return the best-matching document type by keywords, or "Other".

    `min_score` can be lowered for short texts (index.py's auto-route uses this
    on user questions, where a single word like "packaging" is a useful hint).
    """
    scores = score_text(text, header_chars)
    best_type = max(scores, key=scores.get)  # ties go to the type listed first in the dict
    return best_type if scores[best_type] >= min_score else OTHER


def _is_close_call(text: str) -> bool:
    """True if the top two keyword scores are too close to trust."""
    top, runner_up = sorted(score_text(text, CLASSIFIER_HEADER_CHARS).values(), reverse=True)[:2]
    return runner_up >= MIN_SCORE and top - runner_up <= CLOSE_CALL_MARGIN


def classify_with_llm(text: str) -> str | None:
    """Ask the LLM to label a page from its first lines. Returns a label, or None on failure."""
    labels = list(DOC_TYPE_RULES) + [OTHER]
    prompt = (
        "You label pages from a pharmaceutical supplier documentation (SDF) packet.\n"
        f"Reply with exactly one label from this list, and nothing else: {', '.join(labels)}.\n"
        "Use 'Other' if no label fits.\n\n"
        f"Start of the page:\n{text[:800]}"
    )
    reply = call_llm(prompt)
    if reply.startswith(ERROR_PREFIX):
        return None  # LLM unavailable: keep the keyword result rather than fail the upload
    # Match the reply to a label; check longer labels first so "Other" can't shadow them.
    reply = reply.strip().strip(".").lower()
    for label in sorted(labels, key=len, reverse=True):
        if label.lower() in reply:
            return label
    return None


def split_into_documents(pages: list[dict], use_llm: bool = USE_LLM_CLASSIFIER) -> list[dict]:
    """Classify each page and merge consecutive pages of the same type (and file).

    Returns one dict per document:
        {"doc_type", "source", "page_start", "page_end", "ocr_used", "pages"}
    `pages` keeps the original page dicts so chunking.py can track exact page numbers.
    """
    documents = []
    for page in pages:
        text = page["text"]
        current = documents[-1] if documents else None
        header = _normalize(text[:CLASSIFIER_HEADER_CHARS])

        if current and current["source"] == page["source"] and "continued" in header:
            # "QUALITY CERTIFICATE ... (continued)": same document as the page before.
            doc_type, method = current["doc_type"], "continued"
        else:
            doc_type, method = classify_page(text, header_chars=CLASSIFIER_HEADER_CHARS), "keywords"
            if use_llm and (doc_type == OTHER or _is_close_call(text)):
                llm_type = classify_with_llm(text)
                if llm_type:
                    doc_type, method = llm_type, "LLM"
        print(f"Page {page['page_num']}: {doc_type} ({method})")

        if current and current["doc_type"] == doc_type and current["source"] == page["source"]:
            # Same type as the previous page -> it's a continuation of that document.
            current["page_end"] = page["page_num"]
            current["ocr_used"] = current["ocr_used"] or page["ocr_used"]
            current["pages"].append(page)
        else:
            documents.append({
                "doc_type": doc_type,
                "source": page["source"],
                "page_start": page["page_num"],
                "page_end": page["page_num"],
                "ocr_used": page["ocr_used"],
                "pages": [page],
            })
    return documents


if __name__ == "__main__":
    # Quick test: `python src/classify.py` (sample PDF) or `python src/classify.py data/other.pdf`
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE_PDF
    pages = load_pdf(pdf_path)

    print("\n=== Keyword scores (title area counts double) ===")
    for page in pages:
        scores = {t: s for t, s in score_text(page["text"], CLASSIFIER_HEADER_CHARS).items() if s > 0}
        print(f"Page {page['page_num']}: {classify_page(page['text'], header_chars=CLASSIFIER_HEADER_CHARS):<26} scores={scores}")

    print("\n=== Documents found (with continuation + LLM fallback) ===")
    documents = split_into_documents(pages)
    print()
    for doc in documents:
        pages_label = (f"page {doc['page_start']}" if doc["page_start"] == doc["page_end"]
                       else f"pages {doc['page_start']}-{doc['page_end']}")
        print(f"- {doc['doc_type']:<26} {pages_label}" + ("  (OCR)" if doc["ocr_used"] else ""))

    print("\n=== Auto-route check on test questions (min_score=1) ===")
    for q in ["What is the lot/batch number on the Certificate of Quality?",
              "Is the product free of animal-origin materials?",
              "Who signed the cover letter, and when?",
              "What packaging is used?",
              "What sterilization dose was used?",
              "What is the capital of France?"]:
        print(f"{classify_page(q, min_score=1):<26} <- {q}")
