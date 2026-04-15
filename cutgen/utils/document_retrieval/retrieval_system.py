from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CHUNKS_PATH = Path("chunks.json")
DEFAULT_INDEX_PATH = Path("retrieval.index")
DEFAULT_METADATA_PATH = Path("retrieval_metadata.json")

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NORMALIZE_EMBEDDINGS = True


# ============================================================================
# Core helpers
# ============================================================================

def load_chunks(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError("chunks.json must contain a top-level JSON list.")

    cleaned: List[Dict[str, Any]] = []
    for item in chunks:
        if not isinstance(item, dict):
            continue

        text = str(item.get("text", "")).strip()
        if not text:
            continue

        cleaned.append(item)

    if not cleaned:
        raise ValueError("No non-empty chunks found in the chunks file.")

    return cleaned


def build_text_for_embedding(chunk: Dict[str, Any]) -> str:
    """
    Add lightweight structure to improve retrieval quality.
    """
    title = str(chunk.get("document_title", "")).strip()
    section = str(chunk.get("section", "")).strip()
    text = str(chunk.get("text", "")).strip()

    parts: List[str] = []
    if title:
        parts.append(f"Document: {title}")
    if section:
        parts.append(f"Section: {section}")
    parts.append(text)

    return "\n".join(parts)


def load_model(model_name: str) -> SentenceTransformer:
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
    )

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]

    if NORMALIZE_EMBEDDINGS:
        # Inner product on normalized vectors = cosine similarity
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(embeddings)
    return index


def write_metadata(metadata: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def build_index(
    chunks_path: Path,
    index_path: Path,
    metadata_path: Path,
    model_name: str,
) -> None:
    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    texts = [build_text_for_embedding(chunk) for chunk in chunks]

    model = load_model(model_name)

    print("Embedding chunks...")
    embeddings = embed_texts(model, texts)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print(f"Writing FAISS index to {index_path}")
    faiss.write_index(index, str(index_path))

    print(f"Writing metadata to {metadata_path}")
    write_metadata(chunks, metadata_path)

    print(f"Done. Indexed {index.ntotal} chunks.")


# ============================================================================
# Retrieval
# ============================================================================

class Retriever:
    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        model_name: str,
    ) -> None:
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.index = faiss.read_index(str(index_path))

        with metadata_path.open("r", encoding="utf-8") as f:
            self.metadata: List[Dict[str, Any]] = json.load(f)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Index size ({self.index.ntotal}) and metadata size ({len(self.metadata)}) do not match."
            )

        self.model = load_model(model_name)

    def _embed_query(self, query: str) -> np.ndarray:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
        )

        if vec.dtype != np.float32:
            vec = vec.astype(np.float32)

        return vec

    def search(
        self,
        query: str,
        top_k: int = 5,
        required_tags: Optional[List[str]] = None,
        preferred_source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        top_k = max(1, top_k)
        fetch_k = min(max(top_k * 5, 20), len(self.metadata))

        query_vec = self._embed_query(query)
        scores, indices = self.index.search(query_vec, fetch_k)

        results: List[Dict[str, Any]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            item = dict(self.metadata[idx])
            item["score"] = float(score)

            item_tags = set(item.get("tags", []))
            if required_tags and not set(required_tags).issubset(item_tags):
                continue

            if preferred_source is not None and item.get("source") == preferred_source:
                item["score"] += 0.05

            results.append(item)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


def pretty_print_results(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("No results.")
        return

    for rank, item in enumerate(results, start=1):
        print("=" * 100)
        print(f"Rank: {rank}")
        print(f"Score: {item.get('score', 0.0):.4f}")
        print(f"Source: {item.get('source', '')}")
        print(f"Section: {item.get('section', '')}")
        print(f"Tags: {item.get('tags', [])}")
        print(f"ID: {item.get('id', '')}")
        print("-" * 100)
        print(item.get("text", ""))
        print()


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and query a local FAISS retrieval index for your chunked corpus."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build embeddings and FAISS index.")
    build_parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to chunks JSON file.",
    )
    build_parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Path to write FAISS index.",
    )
    build_parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to write metadata JSON.",
    )
    build_parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name.",
    )

    query_parser = subparsers.add_parser("query", help="Query the retrieval index.")
    query_parser.add_argument(
        "query",
        type=str,
        help="Search query string.",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return.",
    )
    query_parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Path to FAISS index.",
    )
    query_parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to metadata JSON.",
    )
    query_parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name.",
    )
    query_parser.add_argument(
        "--required-tags",
        nargs="*",
        default=None,
        help="Require all of these tags to appear in returned chunks.",
    )
    query_parser.add_argument(
        "--preferred-source",
        type=str,
        default=None,
        help="Soft preference for a specific source file.",
    )

    demo_parser = subparsers.add_parser("demo", help="Run a few built-in test queries.")
    demo_parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Path to FAISS index.",
    )
    demo_parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to metadata JSON.",
    )
    demo_parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name.",
    )

    return parser.parse_args()


def run_demo(index_path: Path, metadata_path: Path, model_name: str) -> None:
    retriever = Retriever(
        index_path=index_path,
        metadata_path=metadata_path,
        model_name=model_name,
    )

    test_queries = [
        "how to handle out of bounds loads when tiling does not divide evenly",
        "how cute gemm mainloop stages global memory to shared memory",
        "what tiled mma means in cute",
        "how tensor slicing works with underscore",
        "difference between copy and copy_if",
        "how coalesce simplifies layouts",
    ]

    for query in test_queries:
        print("\n" + "#" * 100)
        print(f"QUERY: {query}")
        print("#" * 100 + "\n")
        results = retriever.search(query=query, top_k=5)
        pretty_print_results(results)


def main() -> None:
    args = parse_args()

    if args.command == "build":
        build_index(
            chunks_path=args.chunks,
            index_path=args.index,
            metadata_path=args.metadata,
            model_name=args.model,
        )
        return

    if args.command == "query":
        retriever = Retriever(
            index_path=args.index,
            metadata_path=args.metadata,
            model_name=args.model,
        )
        results = retriever.search(
            query=args.query,
            top_k=args.top_k,
            required_tags=args.required_tags,
            preferred_source=args.preferred_source,
        )
        pretty_print_results(results)
        return

    if args.command == "demo":
        run_demo(
            index_path=args.index,
            metadata_path=args.metadata,
            model_name=args.model,
        )
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()