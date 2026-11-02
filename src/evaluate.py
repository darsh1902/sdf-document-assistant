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
    from src.config import DEFAULT_MODEL, PROJECT_ROOT
    from src.index import DocumentIndex
    from src.rag import answer_question
except ImportError:
    from chunking import process_pdf
    from config import DEFAULT_MODEL, PROJECT_ROOT
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


def run_eval(model_name: str = DEFAULT_MODEL) -> list[dict]:
    """Run every question on every PDF and return one result row per question."""
    rows = []
    for test in json.loads(QUESTIONS_FILE.read_text()):
        pdf = PROJECT_ROOT / test["pdf"]
        print(f"\n=== {pdf.name} ===")
        # The pipeline prints progress for every page; keep the eval output readable.
        with contextlib.redirect_stdout(io.StringIO()):
            chunks, _ = process_pdf(pdf)
            index = DocumentIndex()
            index.build(chunks)

        for question in test["questions"]:
            with contextlib.redirect_stdout(io.StringIO()):
                result = answer_question(index, question["q"], model_name=model_name)
            row = {
                "pdf": pdf.name,
                "question": question["q"],
                "correct": grade_answer(result["answer"], question),
                "right_doc": grade_retrieval(result["sources"], question),
                "confidence": result["confidence"],
                "time": result["response_time"],
                "answer": result["answer"],
            }
            rows.append(row)
            mark = "PASS" if row["correct"] else "FAIL"
            doc = {True: "yes", False: "NO", None: "-"}[row["right_doc"]]
            print(f"[{mark}] doc={doc:<3} conf={row['confidence']:.2f} {row['time']:>5}s  {question['q']}")
            if not row["correct"]:
                print(f"       answer: {result['answer'][:300]}")
    return rows


def write_report(rows: list[dict], model_name: str) -> str:
    """Write a Markdown report (for notes.md / the Slides) and return its text."""
    correct = sum(r["correct"] for r in rows)
    graded_docs = [r for r in rows if r["right_doc"] is not None]
    right_docs = sum(r["right_doc"] for r in graded_docs)
    lines = [
        f"# Evaluation results ({date.today().isoformat()})",
        "",
        f"Model: {model_name}",
        f"**Answers correct: {correct}/{len(rows)}** · "
        f"**Right document retrieved: {right_docs}/{len(graded_docs)}**",
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
    # Optional: python src/evaluate.py "<model name from config.MODELS>"
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    start = time.perf_counter()
    rows = run_eval(model)
    report = write_report(rows, model)
    print("\n" + report.splitlines()[3])
    print(f"Saved to {RESULTS_FILE.name} ({time.perf_counter() - start:.0f}s total)")
