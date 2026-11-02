"""
Chunking with LlamaIndex: turn each classified page into a LlamaIndex `Document`, then split
it into small overlapping `TextNode`s ("chunks") with `SentenceSplitter`.

Small chunks make embeddings precise (one chunk ≈ one topic); the overlap makes sure a
fact sitting on a chunk boundary (e.g. "Lot Number:" | "18356721") appears whole in at
least one chunk. Every node carries metadata (document type, page, file) for citations.
"""

import sys
import time
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

try:
    from src.classify import split_into_documents
    from src.config import CHUNK_OVERLAP, CHUNK_SIZE, SAMPLE_PDF
    from src.ingest import load_pdf
except ImportError:
    from classify import split_into_documents
    from config import CHUNK_OVERLAP, CHUNK_SIZE, SAMPLE_PDF
    from ingest import load_pdf

# How a node's metadata appears in front of its text. LlamaIndex builds two versions:
# - for the embedding: only "doc_type", so "Who signed the cover letter?" matches every
#   chunk of the cover letter, even the ones that never say "cover letter";
# - for the LLM: only "citation", e.g. "[Source: Cover Letter, Page 1, file.pdf]".
TEXT_TEMPLATE = "[Source: {metadata_str}]\n{content}"
METADATA_TEMPLATE = "{value}"
ALL_KEYS = ["doc_type", "page", "source", "ocr_used", "citation"]
EMBED_KEYS = ["doc_type"]
LLM_KEYS = ["citation"]

_splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def page_to_document(page: dict, doc_type: str) -> Document:
    """One LlamaIndex Document per page, with the metadata every chunk will inherit."""
    citation = f"{doc_type}, Page {page['page_num']}, {page['source']}"
    return Document(
        text=page["text"],
        metadata={"doc_type": doc_type, "page": page["page_num"], "source": page["source"],
                  "ocr_used": page["ocr_used"], "citation": citation},
        excluded_embed_metadata_keys=[k for k in ALL_KEYS if k not in EMBED_KEYS],
        excluded_llm_metadata_keys=[k for k in ALL_KEYS if k not in LLM_KEYS],
        text_template=TEXT_TEMPLATE,
        metadata_template=METADATA_TEMPLATE,
    )


def chunk_documents(documents: list[dict]) -> list[TextNode]:
    """Turn the documents from classify.split_into_documents() into LlamaIndex nodes."""
    nodes = []
    for doc in documents:
        # Chunk page by page: chunks never cross a page break. In SDF packets a new
        # page often starts a new document (e.g. two Certificates of Quality for two
        # products), and mixing them in one chunk could blend their lot numbers.
        for page in doc["pages"]:
            for node in _splitter.get_nodes_from_documents([page_to_document(page, doc["doc_type"])]):
                # Stable, readable IDs that are unique across files: "file.pdf#3".
                # (Re-uploading the same PDF gives the same IDs, so index.py can skip duplicates.)
                node.id_ = f"{page['source']}#{len(nodes)}"
                nodes.append(node)
    return nodes


def process_pdf(path: str | Path) -> tuple[list[TextNode], dict]:
    """Full pipeline for one PDF: load -> classify -> chunk.

    Returns (nodes, summary). The summary feeds the Document Info panel in the UI.
    """
    path = Path(path)
    t0 = time.perf_counter()

    pages = load_pdf(path)
    documents = split_into_documents(pages)
    nodes = chunk_documents(documents)

    # Per-type breakdown, e.g. {"Certificate of Quality": {"pages": 2, "chunks": 5}}
    breakdown = {}
    for doc in documents:
        entry = breakdown.setdefault(doc["doc_type"], {"pages": 0, "chunks": 0})
        entry["pages"] += doc["page_end"] - doc["page_start"] + 1
    for node in nodes:
        breakdown[node.metadata["doc_type"]]["chunks"] += 1

    summary = {
        "file_name": path.name,
        "page_count": len(pages),
        "ocr_pages": sum(p["ocr_used"] for p in pages),
        "documents_found": len(documents),
        "documents": [{"doc_type": d["doc_type"], "page_start": d["page_start"],
                       "page_end": d["page_end"]} for d in documents],
        "chunks_created": len(nodes),
        "doc_types": list(breakdown),  # unique types, in order of appearance
        "breakdown": breakdown,
        "processing_time": round(time.perf_counter() - t0, 2),
    }
    print(f"Created {len(nodes)} chunks from {len(documents)} documents "
          f"in {summary['processing_time']}s")
    return nodes, summary


if __name__ == "__main__":
    # Quick test: `python src/chunking.py` (sample PDF) or `python src/chunking.py data/other.pdf`
    from llama_index.core.schema import MetadataMode

    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE_PDF
    nodes, summary = process_pdf(pdf_path)

    print("\n=== Summary ===")
    for key, value in summary.items():
        if key not in ("breakdown", "documents"):
            print(f"{key:<16} {value}")
    print("Breakdown:")
    for doc_type, counts in summary["breakdown"].items():
        print(f"  {doc_type:<26} {counts['pages']} pages, {counts['chunks']} chunks")

    print("\n=== Chunks ===")
    for n in nodes:
        meta = n.metadata
        print(f"\n[{n.id_}] {meta['doc_type']}, p{meta['page']}, {len(n.text)} chars"
              + (", OCR" if meta["ocr_used"] else ""))
        print("  " + n.text[:160].replace("\n", " | ") + ("..." if len(n.text) > 160 else ""))

    # Show exactly what the embedding model and the LLM see for the first chunk.
    print("\n=== What each side sees (first chunk) ===")
    print("EMBED:", nodes[0].get_content(MetadataMode.EMBED)[:120].replace("\n", " | "))
    print("LLM:  ", nodes[0].get_content(MetadataMode.LLM)[:120].replace("\n", " | "))
    print(f"\nOK: {len(nodes)} chunks, avg {sum(len(n.text) for n in nodes) // max(len(nodes), 1)} chars")
