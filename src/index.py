"""
Vector index with LlamaIndex: a `VectorStoreIndex` on a FAISS vector store, embedded with
`HuggingFaceEmbedding` (BAAI/bge-small-en-v1.5).

Embeddings are normalized, so the inner product FAISS computes (IndexFlatIP) equals
cosine similarity: 1.0 = same meaning, ~0 = unrelated.
"""

import sys
from pathlib import Path

# Import order matters on macOS: torch (via sentence_transformers) and faiss each ship
# their own OpenMP library. If faiss loads first, embedding on CPU crashes with a
# segfault (exit code 139). Loading torch first avoids it — don't re-sort these.
import sentence_transformers  # noqa: F401  (imported only to load torch before faiss)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import faiss
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.vector_stores.faiss import FaissVectorStore

try:
    from src.chunking import process_pdf
    from src.classify import OTHER, classify_page
    from src.config import (DEFAULT_K, EMBEDDING_MODEL, RERANK_CANDIDATES, RERANK_MODEL,
                            SAMPLE_PDF, USE_RERANKER)
except ImportError:
    from chunking import process_pdf
    from classify import OTHER, classify_page
    from config import (DEFAULT_K, EMBEDDING_MODEL, RERANK_CANDIDATES, RERANK_MODEL,
                        SAMPLE_PDF, USE_RERANKER)

# BGE models retrieve better when *questions* (not documents) get this prefix.
# See the BAAI/bge-small-en-v1.5 model card.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBEDDING_DIM = 384  # vector size of bge-small-en-v1.5

# The model is shared by every DocumentIndex (one per UI session), so it's
# loaded only once per process instead of once per user or per query.
_embed_model: HuggingFaceEmbedding | None = None


def get_model() -> HuggingFaceEmbedding:
    """Load the embedding model the first time it's needed, then reuse it."""
    global _embed_model
    if _embed_model is None:
        print(f"Loading embedding model {EMBEDDING_MODEL} (first run downloads ~130 MB)...")
        _embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL,
            query_instruction=QUERY_PREFIX,
            normalize=True,  # normalized vectors -> FAISS inner product = cosine similarity
            # CPU on purpose: on a Mac the default is the GPU ("MPS"), which can run out of
            # memory when other apps use it. This small model is fast enough on CPU.
            device="cpu",
        )
    return _embed_model


# Like the embedding model: loaded once, shared by every session.
_reranker: SentenceTransformerRerank | None = None


def get_reranker() -> SentenceTransformerRerank:
    """Load the cross-encoder reranker the first time it's needed, then reuse it."""
    global _reranker
    if _reranker is None:
        print(f"Loading reranker {RERANK_MODEL} (first run downloads it)...")
        # top_n is set per call in rerank(); CPU for the same reason as the embedding model.
        _reranker = SentenceTransformerRerank(model=RERANK_MODEL, top_n=DEFAULT_K, device="cpu")
    return _reranker


class DocumentIndex:
    """Holds the chunks of all uploaded PDFs and finds the ones relevant to a question."""

    def __init__(self, use_reranker: bool = USE_RERANKER):
        self.nodes: list[TextNode] = []
        self.index: VectorStoreIndex | None = None
        self.use_reranker = use_reranker
        # The doc type the last retrieve() auto-routed to (None = searched everything).
        # rag.py reads this so the UI can show "Routed to: Certificate of Quality".
        self.last_routed_type: str | None = None

    def build(self, nodes: list[TextNode]) -> None:
        """Start a fresh index from these nodes (drops anything indexed before)."""
        self.nodes = []
        self.index = None
        self.add(nodes)

    def add(self, nodes: list[TextNode]) -> None:
        """Add more nodes, e.g. from another uploaded PDF."""
        # Skip nodes we already have, so uploading the same PDF twice doesn't
        # duplicate every search result.
        known_ids = {n.node_id for n in self.nodes}
        new_nodes = [n for n in nodes if n.node_id not in known_ids]
        if not new_nodes:
            print("No new chunks to add.")
            return

        if self.index is None:
            vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatIP(EMBEDDING_DIM))
            storage = StorageContext.from_defaults(vector_store=vector_store)
            self.index = VectorStoreIndex(nodes=[], storage_context=storage, embed_model=get_model())
        # LlamaIndex embeds each node's EMBED view: "[Source: Cover Letter]\n<text>" (see
        # chunking.py). A cover letter rarely says "cover letter", so without the label the
        # date chunk of a letter didn't match "Who signed the cover letter, and when?".
        self.index.insert_nodes(new_nodes)
        self.nodes.extend(new_nodes)
        print(f"Indexed {len(new_nodes)} chunks (total: {len(self.nodes)})")

    @property
    def doc_types(self) -> list[str]:
        """Document types present in the index, for the UI filter dropdown."""
        return list(dict.fromkeys(n.metadata["doc_type"] for n in self.nodes))  # unique, keeps order

    def _search(self, query: QueryBundle, k: int, doc_type: str | None = None) -> list[NodeWithScore]:
        """Top-k nodes for a query, optionally only of one doc type."""
        # LlamaIndex's FAISS store can't filter by metadata ("not implemented for Faiss"),
        # so with a filter we rank ALL nodes and keep the matching ones. A smaller pool
        # (e.g. k*5) can miss a type completely: on a 10-page PDF the cover letter wasn't
        # in the top 15 for "Who signed the cover letter?". IndexFlatIP compares against
        # every vector anyway, so this costs nothing extra.
        pool = len(self.nodes) if doc_type else min(k, len(self.nodes))
        results = self.index.as_retriever(similarity_top_k=pool).retrieve(query)
        if doc_type:
            results = [r for r in results if r.node.metadata["doc_type"] == doc_type]
        return results[:k]

    def retrieve(self, query: str, k: int = DEFAULT_K, filter_doc_type: str | None = None,
                 auto_route: bool = True) -> list[NodeWithScore]:
        """Return the k most relevant nodes (each with its cosine similarity .score).

        - filter_doc_type: only search this document type ("All" or None = everything).
        - auto_route: if no filter is set, guess the doc type from the question's
          keywords and give that type half the result slots (soft routing, see below).
        - With the reranker on, vector search gathers RERANK_CANDIDATES nodes and the
          reranker picks the best k; the order is then the reranker's.
        """
        self.last_routed_type = None
        if not self.nodes or not query.strip():
            return []

        # Embed the question once and reuse it for every search below.
        bundle = QueryBundle(query_str=query, embedding=get_model().get_query_embedding(query))
        # With reranking, cast a wider net first; the reranker narrows it down to k.
        pool = max(k, RERANK_CANDIDATES) if self.use_reranker else k

        # 1. The user picked a type in the dropdown: respect it, no fallback.
        if filter_doc_type and filter_doc_type != "All":
            return self._rerank(bundle, self._search(bundle, pool, filter_doc_type), k)

        # 2. Best matches across all documents (also the final answer when nothing is routed).
        overall = self._search(bundle, pool)

        # 3. Soft auto-route: one keyword in a short question is enough of a hint (min_score=1).
        # The guessed type gets half the slots and the best overall matches get the rest.
        # Searching ONLY the guessed type was fragile: one mislabelled page (e.g. a
        # sterilization certificate tagged as the Certificate of Quality) hid the real answer.
        if auto_route:
            guessed = classify_page(query, min_score=1)
            if guessed != OTHER and guessed in self.doc_types:
                routed = self._search(bundle, pool, guessed)
                if routed:
                    self.last_routed_type = guessed
                    print(f"Auto-routed to: {guessed} (soft)")
                    return self._rerank(bundle, self._merge(routed, overall, pool), k)

        return self._rerank(bundle, overall, k)

    def _rerank(self, query: QueryBundle, candidates: list[NodeWithScore], k: int) -> list[NodeWithScore]:
        """Let the cross-encoder pick the best k candidates (or just take the top k if it's off)."""
        if not self.use_reranker or len(candidates) <= 1:
            return candidates[:k]
        reranker = get_reranker()
        reranker.top_n = k
        # The reranker reads each node's EMBED view, so it also sees "[Source: Cover Letter]"
        # (see chunking.py): a letter's signature chunk never says "cover letter" itself.
        # It replaces .score with its own score, so copies are passed in, and the cosine
        # scores are put back afterwards: the UI's confidence and Sources stay 0-1 cosine values.
        cosine = {c.node.node_id: c.score for c in candidates}
        reranked = reranker.postprocess_nodes(
            [NodeWithScore(node=c.node, score=c.score) for c in candidates], query_bundle=query)
        return [NodeWithScore(node=r.node, score=cosine[r.node.node_id]) for r in reranked]

    @staticmethod
    def _merge(routed: list[NodeWithScore], overall: list[NodeWithScore], k: int) -> list[NodeWithScore]:
        """Top half from the routed type, then fill with the best overall nodes; best first."""
        n_routed = (k + 1) // 2
        picked = routed[:n_routed]
        seen = {r.node.node_id for r in picked}
        # Fill the rest with the best overall matches; if those run out, use more routed ones.
        for result in overall + routed[n_routed:]:
            if len(picked) == k:
                break
            if result.node.node_id not in seen:
                picked.append(result)
                seen.add(result.node.node_id)
        return sorted(picked, key=lambda r: r.score, reverse=True)


if __name__ == "__main__":
    # Quick test: `python src/index.py` (sample PDF) or `python src/index.py data/other.pdf`
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE_PDF
    nodes, summary = process_pdf(pdf_path)

    doc_index = DocumentIndex()
    doc_index.build(nodes)
    print(f"Doc types in index: {doc_index.doc_types}")

    def show(question: str, **kwargs) -> None:
        results = doc_index.retrieve(question, k=3, **kwargs)
        extra = f"  {kwargs}" if kwargs else ""
        print(f"\nQ: {question}{extra}")
        print(f"   auto-routed to: {doc_index.last_routed_type or '-'}")
        for r in results:
            preview = r.node.text[:90].replace("\n", " | ")
            print(f"   {r.score:.2f}  {r.node.metadata['doc_type']}, p{r.node.metadata['page']}: {preview}...")

    print("\n=== Test questions ===")
    for q in ["What is the lot/batch number on the Certificate of Quality?",
              "What is the expiry or retest date?",
              "Is the product free of animal-origin materials?",
              "Who signed the cover letter, and when?",
              "What packaging is used?",
              "What is the recommended storage temperature?",
              "What is the capital of France?"]:
        show(q)

    print("\n=== Filter + auto-route off ===")
    show("What is the storage temperature?", filter_doc_type="Cover Letter")
    show("What is the lot number?", auto_route=False)

    print("\n=== Adding the same PDF again (should skip duplicates) ===")
    doc_index.add(nodes)
