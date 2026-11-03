"""
Evaluation: run a fixed set of questions with known answers on several SDF PDFs and score
the pipeline, so changes are measured instead of eyeballed.

Questions live in data/eval_questions.json. For each question we check:
- Answer correct: the answer contains every expected value (or, for off-topic questions,
  says the documents don't contain it).
- Right doc: at least one retrieved source has the expected document type and page.

Run:  python src/evaluate.py   (results also go to eval_results.md for notes and slides)
"""

import contextlib
import io
import json
import sys
import time
from datetime import date

try:
    from src.chunking import process_pdf
    from src.config import DEFAULT_K, DEFAULT_MODEL, PROJECT_ROOT, RERANK_MODEL, USE_RERANKER
    from src.index import DocumentIndex
    from src.rag import answer_question
except ImportError:
    from chunking import process_pdf
    from config import DEFAULT_K, DEFAULT_MODEL, PROJECT_ROOT, RERANK_MODEL, USE_RERANKER
    from index import DocumentIndex
    from rag import answer_question

QUESTIONS_FILE = PROJECT_ROOT / "data" / "eval_questions.json"
RESULTS_FILE = PROJECT_ROOT / "eval_results.md"
NOT_FOUND_PHRASES = ["don't contain", "do not contain", "not found", "couldn't find"]


def grade_answer(answer: str, question: dict) -> bool:
    """True if the answer has every expected value (case-insensitive), or correctly says 'not found'."""
    text = answer.lower().replace("’", "'")  # some models use a curly apostrophe
    if question.get("not_found"):
        return any(phrase in text for phrase in NOT_FOUND_PHRASES)
    return all(value.lower() in text for value in question["must_contain"])


def grade_retrieval(sources: list[dict], question: dict) -> bool | None:
    """True if a source has the expected doc type and page. None for off-topic questions."""
    if question.get("not_found"):
        return None
    return any(s["doc_type"] == question["doc_type"]
               and any(s["page_start"] <= p <= s["page_end"] for p in question["pages"])
               for s in sources)


def ranking_metrics(results: list, question: dict) -> dict | None:
    """Standard retrieval metrics for one question; None for off-topic questions.

    A chunk is "relevant" if its doc type and page match the expected answer location.
    - hit:       at least one relevant chunk was retrieved
    - rr:        reciprocal rank, 1 / position of the first relevant chunk (MRR = average)
    - precision: share of the retrieved chunks that are relevant
    - recall:    share of the expected pages that were retrieved
    """
    if question.get("not_found"):
        return None
    relevant = [r.node.metadata["doc_type"] == question["doc_type"]
                and r.node.metadata["page"] in question["pages"] for r in results]
    found_pages = {r.node.metadata["page"] for r, rel in zip(results, relevant) if rel}
    return {
        "hit": any(relevant),
        "rr": next((1 / (i + 1) for i, rel in enumerate(relevant) if rel), 0.0),
        "precision": sum(relevant) / len(results) if results else 0.0,
        "recall": len(found_pages) / len(question["pages"]),
    }


def run_eval(model_name: str = DEFAULT_MODEL, use_reranker: bool = USE_RERANKER) -> list[dict]:
    """Run every question on every PDF and return one result row per question."""
    rows = []
    for test in json.loads(QUESTIONS_FILE.read_text()):
        pdf = PROJECT_ROOT / test["pdf"]
        print(f"\n=== {pdf.name} ===")
        # The pipeline prints progress for every page; keep the eval output readable.
        with contextlib.redirect_stdout(io.StringIO()):
            chunks, _ = process_pdf(pdf)
            index = DocumentIndex(use_reranker=use_reranker)
            index.build(chunks)

        for question in test["questions"]:
            with contextlib.redirect_stdout(io.StringIO()):
                start = time.perf_counter()
                retrieved = index.retrieve(question["q"])  # same chunks answer_question gets
                retrieval_ms = (time.perf_counter() - start) * 1000
                result = answer_question(index, question["q"], model_name=model_name)
            row = {
                "pdf": pdf.name,
                "question": question["q"],
                "correct": grade_answer(result["answer"], question),
                "right_doc": grade_retrieval(result["sources"], question),
                "confidence": result["confidence"],
                "time": result["response_time"],
                "retrieval_ms": retrieval_ms,
                "ranking": ranking_metrics(retrieved, question),
                "answer": result["answer"],
            }
            rows.append(row)
            mark = "PASS" if row["correct"] else "FAIL"
            doc = {True: "yes", False: "NO", None: "-"}[row["right_doc"]]
            print(f"[{mark}] doc={doc:<3} conf={row['confidence']:.2f} {row['time']:>5}s  {question['q']}")
            if not row["correct"]:
                print(f"       answer: {result['answer'][:300]}")
    return rows


def write_report(rows: list[dict], model_name: str, use_reranker: bool = USE_RERANKER) -> str:
    """Write a Markdown report (for notes.md / the Slides) and return its text."""
    correct = sum(r["correct"] for r in rows)
    graded_docs = [r for r in rows if r["right_doc"] is not None]
    right_docs = sum(r["right_doc"] for r in graded_docs)
    ranked = [r["ranking"] for r in rows if r["ranking"]]
    avg = lambda key: sum(m[key] for m in ranked) / len(ranked)  # noqa: E731
    retrieval_ms = sum(r["retrieval_ms"] for r in rows) / len(rows)
    lines = [
        f"# Evaluation results ({date.today().isoformat()})",
        "",
        f"Model: {model_name} · Reranker: {RERANK_MODEL if use_reranker else 'off'} · k = {DEFAULT_K}",
        f"**Answers correct: {correct}/{len(rows)}** · "
        f"**Right document retrieved: {right_docs}/{len(graded_docs)}**",
        "",
        f"Retrieval on {len(ranked)} answerable questions: "
        f"Hit rate {avg('hit'):.0%} · MRR {avg('rr'):.2f} · "
        f"Precision@{DEFAULT_K} {avg('precision'):.1%} · Recall@{DEFAULT_K} {avg('recall'):.1%} · "
        f"retrieval {retrieval_ms:.0f} ms per question",
        "",
        "| PDF | Question | Correct | Right doc | Confidence | Time (s) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        doc = {True: "yes", False: "no", None: "n/a"}[r["right_doc"]]
        lines.append(f"| {r['pdf']} | {r['question']} | {'yes' if r['correct'] else 'no'} | "
                     f"{doc} | {r['confidence']:.2f} | {r['time']} |")
    report = "\n".join(lines) + "\n"
    RESULTS_FILE.write_text(report)
    return report


if __name__ == "__main__":
    # python src/evaluate.py                       -> default model, reranker as in config.py
    # python src/evaluate.py --no-rerank           -> same, reranker off (to compare)
    # python src/evaluate.py "<model name from config.MODELS>"
    args = [a for a in sys.argv[1:] if a != "--no-rerank"]
    rerank = USE_RERANKER and "--no-rerank" not in sys.argv
    model = args[0] if args else DEFAULT_MODEL
    start = time.perf_counter()
    rows = run_eval(model, use_reranker=rerank)
    report = write_report(rows, model, use_reranker=rerank)
    print("\n" + "\n".join(report.splitlines()[2:7]))
    print(f"Saved to {RESULTS_FILE.name} ({time.perf_counter() - start:.0f}s total)")
