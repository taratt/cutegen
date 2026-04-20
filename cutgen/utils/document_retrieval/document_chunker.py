from __future__ import annotations

from pathlib import Path
import json
import re
import uuid
from typing import List, Dict, Any, Tuple


# Folder containing your cleaned files
CORPUS_DIR = Path("/home/tarasaba/PycharmProjects/cutgen/cutgen/corpus")

# Output JSON file
OUTPUT_PATH = Path("/home/tarasaba/PycharmProjects/cutgen/cutgen/corpus/chunks/chunks.json")

# Soft chunk size target
MAX_CHARS = 4000
OVERLAP = 400  # only used in the last-resort fallback splitter
# ---------------------------------------------------------------------
# Canonical specs
# ---------------------------------------------------------------------

FILE_SPECS: Dict[str, Dict[str, Any]] = {
    "advance_programming.txt": {
        "document_title": "Advanced Kernel Programming",
        "tags": ["cuda", "async", "barrier", "pipeline", "ptx", "tma"],
        "mode": "sectioned_auto_numbered",
        "aliases": ["advance_programming(1).txt"],
    },
    "cuda_simt.txt": {
        "document_title": "Writing CUDA SIMT Kernels",
        "tags": ["cuda", "simt", "memory", "occupancy"],
        "mode": "sectioned_auto_numbered",
        "aliases": ["cuda_simt(1).txt"],
    },
    "performance_guidelines.txt": {
        "document_title": "Overall Performance Optimization Strategies",
        "tags": ["cuda", "performance", "memory", "occupancy"],
        "mode": "sectioned_auto_numbered",
        "aliases": ["performance_guidelines(1).txt"],
    },
    "cute_quickstart.txt": {
        "document_title": "Getting Started With CuTe",
        "tags": ["cute", "quickstart", "layout", "tensor"],
        "mode": "sectioned_known_headers",
        "aliases": ["cute_quickstart(3).txt"],
        "sections": [
            "Getting Started With CuTe",
            "Library Organization",
            "Quick Tips",
            "How do I print CuTe objects on host or device?",
            "Other output formats",
        ],
    },
    "cute_layouts.txt": {
        "document_title": "CuTe Layouts",
        "tags": ["cute", "layout"],
        "mode": "sectioned_known_headers",
        "aliases": ["cute_layouts(3).txt"],
        "sections": [
            "CuTe Layouts",
            "Fundamental Types and Concepts",
            "Integers",
            "Tuple",
            "IntTuple",
            "Shapes and Strides",
            "Layout",
            "Tensor",
            "Layout Creation and Use",
            "Hierarchical access functions",
            "Constructing a Layout",
            "Using a Layout",
            "Vector Layouts",
            "Matrix examples",
            "Layout Concepts",
            "Layout compatibility",
            "Layouts Coordinates",
            "Coordinate Mapping",
            "Index Mapping",
            "Layout Manipulation",
            "Sublayouts",
            "Concatenation",
            "Grouping and flattening",
            "Summary",
        ],
    },
    "cute_layout_algebra.txt": {
        "document_title": "CuTe Layout Algebra",
        "tags": ["cute", "layout"],
        "mode": "sectioned_known_headers",
        "aliases": ["cute_layout_algebra(3).txt"],
        "sections": [
            "CuTe Layout Algebra",
            "Coalesce",
            "By-mode Coalesce",
            "Composition",
            "Computing Composition",
            "Example 1 – Worked Example of Calculating a Composition",
            "Example 2 – Reshape a layout into a matrix",
            "Example 3 – Reshape a layout into a matrix",
            "By-mode Composition",
            "Composition Tilers",
            "Complement",
            "Complement Examples",
            "Product (Tiling)",
            "Logical Product 1-D Example",
            "Blocked and Raked Products",
        ],
    },
    "cute_tensors.txt": {
        "document_title": "CuTe Tensors",
        "tags": ["cute", "tensor", "layout"],
        "mode": "sectioned_known_headers",
        "aliases": ["cute_tensors(3).txt"],
        "sections": [
            "CuTe Tensors",
            "Fundamental operations",
            "Tensor Engines",
            "Tagged Iterators",
            "Tensor Creation",
            "Nonowning Tensors",
            "Owning Tensors",
            "Accessing a Tensor",
            "Tiling a Tensor",
            "Slicing a Tensor",
            "Partitioning a Tensor",
            "Inner and outer partitioning",
            "Thread-Value partitioning",
            "Examples",
            "Copy a subtile from global memory to registers",
            "Summary",
        ],
    },
    "cute_tensor_algorithms.txt": {
        "document_title": "CuTe Tensor algorithms",
        "tags": ["cute", "tensor", "algorithm", "copy", "gemm"],
        "mode": "sectioned_known_headers",
        "aliases": ["cute_tensor_algorithms(3).txt"],
        "sections": [
            "CuTe Tensor algorithms",
            "copy",
            "Interface and specialization opportunities",
            "Parallelism and synchronization depend on parameter types",
            "A generic copy implementation",
            "copy_if",
            "gemm",
            "What gemm computes",
            "Dispatch to optimized implementations",
            "axpby",
            "fill",
            "clear",
            "Other algorithms",
        ],
    },
    "cute_mma.txt": {
        "document_title": "CuTe’s support for Matrix Multiply-Accumulate instructions",
        "tags": ["cute", "mma", "optimization"],
        "mode": "sectioned_known_headers",
        "aliases": ["cute_mma(3).txt"],
        "sections": [
            "CuTe’s support for Matrix Multiply-Accumulate instructions",
            "CuTe MMA Atoms",
            "Operation structs",
            "Location of files",
            "Operation struct’s name",
            "Contents",
            "Type aliases",
            "fma static member device function",
            "Traits",
            "Example",
            "Volta",
            "Types",
            "Shape",
            "Thread ID",
            "Accumulator Mapping",
            "A and B Layout Mapping",
            "Hopper",
            "TiledMMAs",
        ],
    },
    "gemm_tutorial.txt": {
        "document_title": "CuTe dense matrix-matrix multiply tutorial",
        "tags": ["cute", "gemm", "mma", "tutorial", "optimization"],
        "mode": "sectioned_known_headers",
        "aliases": ["gemm_tutorial(3).txt"],
        "sections": [
            "CuTe dense matrix-matrix multiply tutorial",
            "sgemm_1.cu",
            "High-level interface",
            "The Full Tensors: Shapes, Strides, and Data",
            "CTA Partitioning",
            "SMEM tensors",
            "Copy partitioning",
            "Math partitioning",
            "Mainloop",
            "sgemm_2.cu",
            "TiledCopy",
            "TiledMMA",
            "Other changes",
            "sgemm_sm70.cu",
            "sgemm_sm80.cu",
            "Next steps",
            "GETT as GEMM",
        ],
    },
    "predication.txt": {
        "document_title": "Predication — What to do when tiling isn’t perfect",
        "tags": ["cute", "predication", "gemm", "layout", "debug"],
        "mode": "sectioned_known_headers",
        "aliases": ["predication(3).txt"],
        "sections": [
            "Predication: What to do when tiling isn’t perfect",
            "Example: Generic vector tiling with predication",
            "General predication procedure",
            "Example: Predicating the GEMM epilogue",
            "Advantages of the coordinate tensor approach",
            "Classic CUDA analogy",
            "Example: m- and n-predication of A and B loads in GEMM",
            "CTA partitioning",
            "Thread partitioning",
            "Create identity coordinate tensors",
            "Apply same partitioning",
            "Create predicate tensors",
            "Populate predicates",
            "Use predicates in copy_if",
        ],
    },
    "efficient_gemm": {
        "document_title": "Efficient GEMM in CUDA",
        "tags": ["cutlass", "gemm", "mma", "optimization"],
        "mode": "sectioned_known_headers",
        "aliases": ["efficient_gemm(1)", "efficient_gemm.txt", "efficient_gemm(1).txt"],
        "sections": [
            "Efficient GEMM in CUDA",
            "Hierarchical Structure",
            "Threadblock-level GEMM",
            "Warp-level GEMM",
            "Thread-level GEMM",
            "Epilogue",
            "Optimizations",
            "Pipelining",
            "Threadblock Rasterization",
            "Parallelized Reductions",
            "Split K - reduction across threadblocks",
            "Sliced K - reduction across warps",
        ],
    },
    "sgemm_sm80_cu.txt": {
        "document_title": "sgemm_sm80.cu",
        "tags": ["code", "gemm", "mma", "sm80", "cutlass"],
        "mode": "full_file",
        "aliases": ["sgemm_sm80_cu(1).txt"],
    },
}


def build_alias_map(specs: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for canonical, spec in specs.items():
        alias_map[canonical] = canonical
        for alias in spec.get("aliases", []):
            alias_map[alias] = canonical
    return alias_map


ALIAS_MAP = build_alias_map(FILE_SPECS)


# ---------------------------------------------------------------------
# Contamination filters
# ---------------------------------------------------------------------

BAD_TEXT_PATTERNS = [
    r"\.\.\.\s*\(continues exactly as provided, unchanged\)",
    r"\.\.\.\s*\(continues.*unchanged\)",
    r"copy-paste-ready",
    r"wrapped into a single",
    r"same rule as before",
    r"Ready for the next file whenever you are",
    r"Below is your content wrapped",
]

BAD_TEXT_REGEXES = [re.compile(p, re.IGNORECASE) for p in BAD_TEXT_PATTERNS]


def contains_contamination(text: str) -> bool:
    return any(rx.search(text) for rx in BAD_TEXT_REGEXES)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def normalize_line(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("’", "").replace("'", "")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "section"


def canonicalize_filename(name: str) -> str:
    if name in ALIAS_MAP:
        return ALIAS_MAP[name]
    stripped = re.sub(r"\(\d+\)(?=(\.[^.]+)?$)", "", name)
    if stripped in ALIAS_MAP:
        return ALIAS_MAP[stripped]
    return name


def strip_title_lines(lines: List[str]) -> List[str]:
    out = []
    for line in lines:
        if line.startswith("TITLE:"):
            continue
        if normalize_line(line) == "CuTe Documentation":
            continue
        out.append(line)
    return out


def is_numbered_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return re.match(r"^\d+(?:\.\d+)*\.\s+\S", s) is not None


def is_code_line(line: str) -> bool:
    raw = line.rstrip("\n")
    s = raw.strip()
    if not s:
        return False

    prefixes = (
        "#include",
        "#define",
        "__global__",
        "__device__",
        "__host__",
        "__shared__",
        "__launch_bounds__",
        "template <",
        "template<",
        "using ",
        "typedef ",
        "struct ",
        "class ",
        "namespace ",
        "return ",
        "if (",
        "for (",
        "while (",
        "switch (",
        "case ",
        "auto ",
        "int ",
        "float ",
        "double ",
        "bool ",
        "Tensor ",
        "Layout ",
        "Thr",
        "CUTE_",
        "cuda",
        "cp_async",
        "extern __shared__",
    )
    if s.startswith(prefixes):
        return True

    if s.startswith("//") or s.startswith("/*") or s.startswith("*") or s.startswith("*/"):
        return True

    if any(tok in s for tok in ("{", "}", ";", "::", "->", "<<<", ">>>")):
        return True

    if raw.startswith(("  ", "\t")) and any(tok in s for tok in ("(", ")", "=", "[", "]", ";")):
        return True

    return False


def split_into_code_aware_blocks(text: str) -> List[Tuple[str, bool]]:
    lines = text.splitlines()
    if not lines:
        return []

    blocks: List[Tuple[str, bool]] = []
    current_lines: List[str] = []
    current_kind: Optional[bool] = None

    def flush() -> None:
        nonlocal current_lines, current_kind
        if current_lines:
            block_text = "\n".join(current_lines).strip()
            if block_text:
                blocks.append((block_text, bool(current_kind)))
        current_lines = []
        current_kind = None

    for line in lines:
        if not line.strip():
            if current_lines:
                current_lines.append(line)
            continue

        line_is_code = is_code_line(line)

        if current_kind is None:
            current_kind = line_is_code
            current_lines = [line]
            continue

        if line_is_code == current_kind:
            current_lines.append(line)
        else:
            flush()
            current_kind = line_is_code
            current_lines = [line]

    flush()
    return blocks


def split_large_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    blocks = split_into_code_aware_blocks(text)
    chunks: List[str] = []
    current = ""

    def emit(chunk_text: str) -> None:
        chunk_text = chunk_text.strip()
        if chunk_text and not contains_contamination(chunk_text):
            chunks.append(chunk_text)

    for block_text, is_code in blocks:
        if len(block_text) <= max_chars:
            candidate = f"{current}\n\n{block_text}".strip() if current else block_text
            if len(candidate) <= max_chars:
                current = candidate
            else:
                emit(current)
                current = block_text
            continue

        if current:
            emit(current)
            current = ""

        if is_code:
            emit(block_text)
            continue

        paragraphs = [p.strip() for p in block_text.split("\n\n") if p.strip()]
        para_current = ""

        for p in paragraphs:
            candidate = f"{para_current}\n\n{p}".strip() if para_current else p
            if len(candidate) <= max_chars:
                para_current = candidate
            else:
                if para_current:
                    emit(para_current)
                    para_current = p
                else:
                    start = 0
                    while start < len(p):
                        end = start + max_chars
                        piece = p[start:end].strip()
                        if piece and not contains_contamination(piece):
                            chunks.append(piece)
                        start += max_chars - overlap
                    para_current = ""

        if para_current:
            emit(para_current)

    if current:
        emit(current)

    return chunks


def dedupe_header_positions(header_positions: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    seen = set()
    for idx, header in header_positions:
        key = (idx, header)
        if key not in seen:
            out.append((idx, header))
            seen.add(key)
    return out


def find_known_header_positions(lines: List[str], headers: List[str]) -> List[Tuple[int, str]]:
    normalized_headers = {normalize_line(h): h for h in headers}
    found: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines):
        norm = normalize_line(line)
        if norm in normalized_headers:
            found.append((idx, normalized_headers[norm]))
    return dedupe_header_positions(found)


def find_numbered_header_positions(lines: List[str]) -> List[Tuple[int, str]]:
    found: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines):
        s = line.strip()
        if is_numbered_heading(s):
            found.append((idx, s))
    return dedupe_header_positions(found)


def build_chunks_from_sections(
    path: Path,
    document_title: str,
    tags: List[str],
    lines: List[str],
    header_positions: List[Tuple[int, str]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []

    for section_idx, (start_idx, section_name) in enumerate(header_positions):
        body_start = start_idx + 1
        body_end = header_positions[section_idx + 1][0] if section_idx + 1 < len(header_positions) else len(lines)
        body = "\n".join(lines[body_start:body_end]).strip()

        if not body:
            continue
        if contains_contamination(body):
            print(f"[WARN] skipping contaminated section in {path.name}: {section_name}")
            continue

        subchunks = split_large_text(body)

        for chunk_idx, chunk in enumerate(subchunks):
            if not chunk.strip():
                continue
            if contains_contamination(chunk):
                print(f"[WARN] skipping contaminated chunk in {path.name}: {section_name} chunk {chunk_idx}")
                continue

            output.append(
                {
                    "id": f"{path.stem}__{slugify(section_name)}__sec{section_idx}__chunk{chunk_idx}",
                    "uuid": str(uuid.uuid4()),
                    "source": path.name,
                    "document_title": document_title,
                    "section": section_name,
                    "section_index_in_file": section_idx,
                    "chunk_index_within_section": chunk_idx,
                    "text": chunk,
                    "tags": list(tags),
                }
            )

    return output


# ---------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------

def process_full_file(path: Path, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if contains_contamination(text):
        print(f"[WARN] skipping contaminated full file: {path.name}")
        return []

    return [
        {
            "id": f"{path.stem}__full_file__sec0__chunk0",
            "uuid": str(uuid.uuid4()),
            "source": path.name,
            "document_title": spec["document_title"],
            "section": "FULL_FILE",
            "section_index_in_file": 0,
            "chunk_index_within_section": 0,
            "text": text,
            "tags": list(spec["tags"]),
        }
    ]


def process_sectioned_known_headers(path: Path, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if contains_contamination(text):
        print(f"[WARN] file contains contamination marker: {path.name}")
    lines = strip_title_lines(text.splitlines())
    header_positions = find_known_header_positions(lines, spec["sections"])

    if not header_positions:
        raise ValueError(f"No known headers found in {path.name}")

    found_set = {h for _, h in header_positions}
    missing = [h for h in spec["sections"] if h not in found_set]
    if missing:
        print(f"[WARN] {path.name}: {len(missing)} missing known headers")
        for h in missing:
            print(f"       - {h}")

    return build_chunks_from_sections(
        path=path,
        document_title=spec["document_title"],
        tags=spec["tags"],
        lines=lines,
        header_positions=header_positions,
    )


def process_sectioned_auto_numbered(path: Path, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if contains_contamination(text):
        print(f"[WARN] file contains contamination marker: {path.name}")
    lines = strip_title_lines(text.splitlines())
    header_positions = find_numbered_header_positions(lines)

    if not header_positions:
        raise ValueError(f"No numbered headers found in {path.name}")

    return build_chunks_from_sections(
        path=path,
        document_title=spec["document_title"],
        tags=spec["tags"],
        lines=lines,
        header_positions=header_positions,
    )


def process_file(path: Path) -> List[Dict[str, Any]]:
    canonical_name = canonicalize_filename(path.name)

    if canonical_name not in FILE_SPECS:
        raise ValueError(f"No spec for file: {path.name}")

    spec = FILE_SPECS[canonical_name]
    mode = spec["mode"]

    if mode == "full_file":
        return process_full_file(path, spec)
    if mode == "sectioned_known_headers":
        return process_sectioned_known_headers(path, spec)
    if mode == "sectioned_auto_numbered":
        return process_sectioned_auto_numbered(path, spec)

    raise ValueError(f"Unknown mode {mode} for {path.name}")


def main() -> None:
    all_chunks: List[Dict[str, Any]] = []
    seen_ids = set()

    for path in sorted(CORPUS_DIR.iterdir()):
        if not path.is_file():
            continue

        canonical_name = canonicalize_filename(path.name)
        if canonical_name not in FILE_SPECS:
            print(f"[INFO] skipping {path.name}")
            continue

        try:
            chunks = process_file(path)
        except Exception as e:
            print(f"[ERROR] {path.name}: {e}")
            continue

        for chunk in chunks:
            if chunk["id"] in seen_ids:
                raise ValueError(f"Duplicate chunk id: {chunk['id']}")
            seen_ids.add(chunk["id"])
            all_chunks.append(chunk)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(all_chunks)} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()