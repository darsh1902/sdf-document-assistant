"""
The full RAG pipeline with LlamaIndex: question -> retrieve nodes (index.py) -> LlamaIndex
response synthesizer with our prompt -> answer + sources + confidence.
"""

import sys
import time
from pathlib import Path

from llama_index.core import PromptTemplate, get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import NodeWithScore

try:
    from src.chunking import process_pdf
    from src.config import DEFAULT_K, DEFAULT_MODEL, SAMPLE_PDF
    from src.index import DocumentIndex
    from src.llm import ERROR_PREFIX, check_model, friendly_error, get_llm
except ImportError:
    from chunking import process_pdf
    from config import DEFAULT_K, DEFAULT_MODEL, SAMPLE_PDF
    from index import DocumentIndex
    from llm import ERROR_PREFIX, check_model, friendly_error, get_llm

NO_INFO_ANSWER = "I couldn't find relevant information to answer your question."

# LlamaIndex fills in {context_str} (the retrieved chunks, each starting with
# "[Source: <doc type>, Page <n>, <file>]", see chunking.py) and {query_str}.
PROMPT_TEMPLATE = PromptTemplate("""You are a document assistant for pharmaceutical supplier documents.
Answer the question using ONLY the context below.

Rules:
- If the answer is not in the context, say: "The documents don't contain this information."
- Do not guess or use outside knowledge.
- If the question asks about a specific document (for example the cover letter or the
  packaging specification) and no source in the context is that type of document, reply
  exactly: "The <document> was not found in the uploaded documents." and nothing else.
  Never answer from a different document.
- If several products, batches, lots or certificates match, list every one, each with its source.
- Cite the source for each fact, like (Certificate of Quality, Page 3).
- Keep the answer short and clear. Use bullet points for lists.
- Copy exact values (dates, lot numbers, specifications) exactly as written.

Context:
{context_str}

Question: {query_str}

Answer:""")


def _unique_sources(results: list[NodeWithScore]) -> list[dict]:
    """One entry per (file, doc type, page), keeping the best score — for the UI's Sources list."""
    sources = {}
    for r in results:
        meta = r.node.metadata
        key = (meta["source"], meta["doc_type"], meta["page"])
        if key not in sources or r.score > sources[key]["score"]:
            sources[key] = {
                "doc_type": meta["doc_type"],
                "pages": f"Page {meta['page']}",
                "page_start": meta["page"],  # chunks never cross a page, so start = end
                "page_end": meta["page"],
                "source": meta["source"],
                "score": round(r.score, 2),
            }
    return list(sources.values())  # still best-first, because results were


def answer_question(index: DocumentIndex, query: str, k: int = DEFAULT_K,
                    filter_doc_type: str | None = None, auto_route: bool = True,
                    model_name: str = DEFAULT_MODEL) -> dict:
    """Answer a question from the indexed documents.

    Returns a dict with: answer, sources, confidence, chunks_used, routed_doc_type,
    model, response_time, and error (True if the LLM call failed).
    """
    result = {
        "answer": NO_INFO_ANSWER,
        "sources": [],
        "confidence": 0.0,
        "chunks_used": 0,
        "routed_doc_type": None,
        "model": model_name,
        "response_time": 0.0,
        "error": False,
    }

    # 1. Retrieve (LlamaIndex retriever + our soft routing, see index.py).
    results = index.retrieve(query, k=k, filter_doc_type=filter_doc_type, auto_route=auto_route)
    result["routed_doc_type"] = index.last_routed_type

    # 2. Nothing found -> no point asking the LLM.
    if not results:
        return result

    # 3. Confidence = average cosine similarity of the chunks we used. It measures how
    # well the *retrieval* matched, not whether the LLM's answer is correct.
    scores = [r.score for r in results]
    result.update({
        "sources": _unique_sources(results),
        "confidence": round(max(0.0, sum(scores) / len(scores)), 2),  # cosine can dip below 0
        "chunks_used": len(results),
    })

    # 4. Answer with LlamaIndex's response synthesizer: it puts the chunks into our
    # prompt's {context_str} and calls the LLM. COMPACT packs them into as few calls as
    # possible (one, for our small chunks).
    problem = check_model(model_name)
    if problem:
        result.update({"answer": problem, "error": True})
        return result
    synthesizer = get_response_synthesizer(llm=get_llm(model_name), text_qa_template=PROMPT_TEMPLATE,
                                           response_mode=ResponseMode.COMPACT)
    start = time.perf_counter()
    try:
        answer = str(synthesizer.synthesize(query, nodes=results)).strip()
    except Exception as error:  # any LLM failure -> friendly message for the UI, not a crash
        print(f"LLM error: {type(error).__name__}: {error}")
        answer = friendly_error(model_name, error)
    seconds = round(time.perf_counter() - start, 2)
    if not answer or answer == "Empty Response":
        # Happens if a reasoning model spent its whole output budget on thinking.
        answer = f"{ERROR_PREFIX}{model_name}: the model returned an empty answer. Try again."
    print(f"LLM {model_name}: answered in {seconds}s")

    # 5. Package it up for the UI.
    result.update({"answer": answer, "response_time": seconds, "error": answer.startswith(ERROR_PREFIX)})
    return result


if __name__ == "__main__":
    # Quick test:
    #   python src/rag.py                                   -> sample PDF, default model
    #   python src/rag.py data/other.pdf                    -> another PDF
    #   python src/rag.py data/other.pdf "GPT-6 Luna (OpenAI)"
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE_PDF
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    nodes, _summary = process_pdf(pdf_path)
    doc_index = DocumentIndex()
    doc_index.build(nodes)

    questions = [
        "What is the lot/batch number on the Certificate of Quality?",
        "What is the expiry or retest date?",
        "Is the product free of animal-origin materials?",
        "Who signed the cover letter, and when?",
        "What packaging is used?",
        "What is the capital of France?",
    ]
    print(f"\n=== {pdf_path.name} | model: {model} ===")
    for q in questions:
        r = answer_question(doc_index, q, model_name=model)
        sources = "; ".join(f"{s['doc_type']} {s['pages']} ({s['score']})" for s in r["sources"])
        print(f"\nQ: {q}")
        print(f"A: {r['answer']}")
        print(f"   Sources: {sources or '-'}")
        print(f"   Confidence: {r['confidence']} · Chunks used: {r['chunks_used']} · "
              f"Routed: {r['routed_doc_type'] or '-'} · {r['response_time']}s"
              + (" · ERROR" if r["error"] else ""))
