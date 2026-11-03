"""
Gradio UI for the SDF Document Assistant.

Run with:  python app.py   (then open the URL it prints, usually http://127.0.0.1:7860)
"""

from pathlib import Path

import gradio as gr

from src.chunking import process_pdf
from src.config import DEFAULT_K, DEFAULT_MODEL, USE_RERANKER
from src.index import DocumentIndex, get_model, get_reranker
from src.llm import available_models
from src.rag import answer_question

EXAMPLE_QUESTIONS = [
    "What is the lot/batch number on the Certificate of Quality?",
    "What is the expiry or retest date?",
    "Is the product free of animal-origin materials?",
    "Who signed the cover letter, and when?",
    "What packaging is used?",
]

NO_PDF_MESSAGE = "📄 Please upload a PDF and click **Process** first — then I can answer questions about it."


# ---------------------------------------------------------------------------
# Formatting helpers (turn pipeline output into Markdown for the UI)
# ---------------------------------------------------------------------------
def format_summary(summary: dict) -> str:
    """Document Info panel section for one processed PDF."""
    ocr = f" ({summary['ocr_pages']} scanned, OCR used)" if summary["ocr_pages"] else ""
    lines = [
        f"#### {summary['file_name']}",
        f"**{summary['page_count']} pages**{ocr} · **{summary['documents_found']} documents** · "
        f"**{summary['chunks_created']} chunks** · {summary['processing_time']}s",
        "",
        "| Document type | Found on | Pages | Chunks |",
        "|---|---|---|---|",
    ]
    for doc_type, counts in summary["breakdown"].items():
        # A type can appear more than once in a PDF, e.g. "p2-3, p7".
        ranges = ", ".join(
            f"p{d['page_start']}" if d["page_start"] == d["page_end"] else f"p{d['page_start']}-{d['page_end']}"
            for d in summary["documents"] if d["doc_type"] == doc_type
        )
        lines.append(f"| {doc_type} | {ranges} | {counts['pages']} | {counts['chunks']} |")
    return "\n".join(lines)


def format_reply(result: dict) -> str:
    """Chat reply: the answer, then Sources, then a small stats line."""
    # LLM errors and "nothing found" are shown as-is: sources/confidence would be meaningless.
    if result["error"] or result["chunks_used"] == 0:
        return result["answer"]
    # The LLM said the answer isn't in the documents: listing "sources" would contradict it.
    # (Normalize the curly apostrophe some models use: don’t -> don't.)
    if "don't contain this information" in result["answer"].replace("’", "'"):
        return f"{result['answer']}\n\n*Model: {result['model'].split(' (')[0]} · {result['response_time']}s*"

    sources = "\n".join(
        f"- {s['doc_type']}, {s['pages']} — {s['source']} ({s['score']:.2f})"
        for s in result["sources"]
    )
    stats = [f"Confidence: {result['confidence']:.2f}", f"Chunks used: {result['chunks_used']}"]
    if result["routed_doc_type"]:
        stats.append(f"Routed to: {result['routed_doc_type']}")
    stats.append(f"Model: {result['model'].split(' (')[0]}")  # "GPT-6 Luna (OpenAI)" -> "GPT-6 Luna"
    stats.append(f"{result['response_time']}s")

    return f"{result['answer']}\n\n**Sources**\n{sources}\n\n*{' · '.join(stats)}*"


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------
def process_files(files: list[str] | None):
    """Build a fresh index from every PDF currently in the upload box."""
    reset_dropdown = gr.update(choices=["All"], value="All")
    if not files:
        return None, "⚠️ Please upload at least one PDF first.", reset_dropdown

    index = DocumentIndex()
    sections = []
    for path in files:
        name = Path(path).name
        try:
            chunks, summary = process_pdf(path)
        except Exception as error:  # e.g. a corrupted or password-protected PDF
            sections.append(f"#### ❌ {name}\nCouldn't read this file: {error}")
            continue
        if not chunks:
            sections.append(f"#### ⚠️ {name}\nNo text found in this file.")
            continue
        index.add(chunks)
        sections.append(format_summary(summary))

    if not index.nodes:
        return None, "\n\n".join(sections), reset_dropdown

    header = f"✅ **{len(files)} file(s) · {len(index.nodes)} chunks indexed** — ready for questions."
    doc_types = gr.update(choices=["All"] + index.doc_types, value="All")
    return index, "\n\n".join([header] + sections), doc_types


def add_user_message(message: str, history: list):
    """Step 1 of sending: show the question in the chat right away and clear the textbox."""
    message = (message or "").strip()
    if not message:
        gr.Warning("Please type a question first.")
        return gr.update(), history, ""
    return "", history + [{"role": "user", "content": message}], message


def add_bot_reply(question: str, history: list, index: DocumentIndex | None,
                  doc_type: str, auto_route: bool, k: float):
    """Step 2 of sending: run the RAG pipeline and append the answer."""
    if not question:  # step 1 rejected an empty question
        return history
    if index is None:
        reply = NO_PDF_MESSAGE
    else:
        result = answer_question(
            index, question, k=int(k),
            filter_doc_type=None if doc_type == "All" else doc_type,
            auto_route=auto_route, model_name=DEFAULT_MODEL,
        )
        reply = format_reply(result)
    return history + [{"role": "assistant", "content": reply}]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="SDF Document Assistant") as demo:
    # Per-session state: each browser tab gets its own documents.
    index_state = gr.State(None)
    pending_question = gr.State("")  # hands the question from step 1 to step 2

    gr.Markdown(
        "# 📑 SDF Document Assistant\n"
        "Upload supplier PDFs (SDF packets) and ask questions. Answers come **only** from your "
        "documents, with sources and a confidence score."
    )
    if DEFAULT_MODEL not in available_models():
        gr.Markdown(f"⚠️ **{DEFAULT_MODEL}** isn't usable — check `OPENAI_API_KEY` in your `.env` file.")

    with gr.Row():
        # ----- Left column: upload, document info, settings -----
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### 📄 Upload")
            file_input = gr.File(file_count="multiple", file_types=[".pdf"], type="filepath",
                                 label="PDF files")
            process_button = gr.Button("Process", variant="primary")

            gr.Markdown("### 📊 Document Info")
            doc_info = gr.Markdown("*No documents processed yet.*")

            gr.Markdown("### ⚙️ Settings")
            doc_type_dropdown = gr.Dropdown(["All"], value="All", label="Document type")
            auto_route_checkbox = gr.Checkbox(
                value=True, label="Auto-route (guess the document type from the question)")
            k_slider = gr.Slider(1, 10, value=DEFAULT_K, step=1, label="Chunks to retrieve")
            gr.Markdown(f"Model: **{DEFAULT_MODEL}**")

        # ----- Right column: chat -----
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat")
            chatbot = gr.Chatbot(height=540, show_label=False,
                                 placeholder="Upload a PDF, click **Process**, then ask a question.")
            with gr.Row():
                question_box = gr.Textbox(placeholder="Ask about lot numbers, expiry dates, declarations…",
                                          show_label=False, scale=5)
                send_button = gr.Button("Send", variant="primary", scale=1)
            clear_button = gr.Button("Clear chat")
            gr.Examples(EXAMPLE_QUESTIONS, inputs=question_box, label="Try asking")

    # ----- Wiring -----
    process_button.click(process_files, inputs=file_input,
                         outputs=[index_state, doc_info, doc_type_dropdown])

    # Enter and the Send button do the same two steps.
    for trigger in (question_box.submit, send_button.click):
        trigger(add_user_message, inputs=[question_box, chatbot],
                outputs=[question_box, chatbot, pending_question]
        ).then(add_bot_reply,
               inputs=[pending_question, chatbot, index_state, doc_type_dropdown,
                       auto_route_checkbox, k_slider],
               outputs=chatbot)

    clear_button.click(lambda: [], outputs=chatbot)


if __name__ == "__main__":
    # Load the models now, so the first "Process" click and first question aren't slow.
    get_model()
    if USE_RERANKER:
        get_reranker()
    print(f"Model: {DEFAULT_MODEL} | usable models: {available_models()}")
    demo.launch(theme=gr.themes.Soft())
