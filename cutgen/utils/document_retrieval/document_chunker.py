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
# Hard-coded corpus specification for the current files only
# mode:
#   - "sectioned": split by known hard-coded headers
#   - "full_file": keep the whole file as a single chunk
# aliases:
#   - acceptable alternate filenames for the same spec
# ---------------------------------------------------------------------

FILE_SPECS: Dict[str, Dict[str, Any]] = {
    "cute_quickstart.txt": {
        "document_title": "Getting Started With CuTe",
        "tags": ["quickstart", "layout", "tensor"],
        "mode": "sectioned",
        "aliases": ["cute_quickstart(2).txt"],
        "sections": [
            "Getting Started With CuTe",
            "System Requirements",
            "Knowledge prerequisites",
            "Building Tests and Examples",
            "Library Organization",
            "Quick Tips",
            "How do I print CuTe objects on host or device?",
            "Other output formats",
        ],
    },
    "cute_layouts.txt": {
        "document_title": "CuTe Layouts",
        "tags": ["layout"],
        "mode": "sectioned",
        "aliases": ["cute_layouts(2).txt"],
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
        "tags": ["layout"],
        "mode": "sectioned",
        "aliases": ["cute_layout_algebra(2).txt"],
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
        "tags": ["tensor", "layout"],
        "mode": "sectioned",
        "aliases": ["cute_tensors(2).txt"],
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
        "tags": ["tensor", "algorithm", "copy", "gemm"],
        "mode": "sectioned",
        "aliases": ["cute_tensor_algorithms(2).txt"],
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
        "tags": ["mma", "optimization"],
        "mode": "sectioned",
        "aliases": ["cute_mma(2).txt"],
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
        "tags": ["gemm", "mma", "copy", "optimization", "tutorial"],
        "mode": "sectioned",
        "aliases": ["gemm_tutorial(2).txt"],
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
        "tags": ["predication", "debug", "gemm", "layout"],
        "mode": "sectioned",
        "aliases": ["predication(2).txt"],
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
    "cuda_simt.txt": {
        "document_title": "Writing CUDA SIMT Kernels",
        "tags": ["cuda", "simt", "memory", "occupancy"],
        "mode": "sectioned",
        "aliases": [],
        "sections": [
            "2.2. Writing CUDA SIMT Kernels",
            "2.2.1. Basics of SIMT",
            "2.2.2. Thread Hierarchy",
            "2.2.3. GPU Device Memory Spaces",
            "2.2.3.1. Global Memory",
            "2.2.3.2. Shared Memory",
            "2.2.3.2.1. Static Allocation of Shared Memory",
            "2.2.3.2.2. Dynamic Allocation of Shared Memory",
            "2.2.3.3. Registers",
            "2.2.3.4. Local Memory",
            "2.2.3.5. Constant Memory",
            "2.2.3.6. Caches",
            "2.2.3.7. Texture and Surface Memory",
            "2.2.3.8. Distributed Shared Memory",
            "2.2.4. Memory Performance",
            "2.2.4.1. Coalesced Global Memory Access",
            "2.2.4.1.1. Matrix Transpose Example Using Global Memory",
            "2.2.4.2. Shared Memory Access Patterns",
            "2.2.4.2.1. Matrix Transpose Example Using Shared Memory",
            "2.2.4.2.2. Shared Memory Bank Conflicts",
            "2.2.5. Atomics",
            "2.2.6. Cooperative Groups",
            "2.2.7. Kernel Launch and Occupancy",
        ],
    },
    "advance_programming.txt": {
        "document_title": "Advanced Kernel Programming",
        "tags": ["cuda", "async", "barrier", "pipeline", "ptx", "tma"],
        "mode": "sectioned",
        "aliases": [],
        "sections": [
            "3.2. Advanced Kernel Programming",
            "3.2.1. Using PTX",
            "cuda::ptx namespace",
            "Inline PTX",
            "3.2.2. Hardware Implementation",
            "3.2.2.1. SIMT Execution Model",
            "3.2.2.1.1. Independent Thread Scheduling",
            "3.2.2.2. Hardware Multithreading",
            "3.2.2.3. Asynchronous Execution Features",
            "3.2.2.3.1. Async Thread and Async Proxy",
            "3.2.3. Thread Scopes",
            "3.2.4. Advanced Synchronization Primitives",
            "3.2.4.1. Scoped Atomics",
            "3.2.4.1.1. Thread Scope and Memory Ordering",
            "3.2.4.1.2. Performance Considerations",
            "3.2.4.2. Asynchronous Barriers",
            "Temporal Splitting of Synchronization",
            "3.2.4.3. Pipelines",
            "3.2.5. Asynchronous Data Copies",
            "3.2.6. Configuring L1/Shared Memory Balance",
        ],
    },
    "performance_guidelines.txt": {
        "document_title": "Overall Performance Optimization Strategies",
        "tags": ["cuda", "performance", "occupancy", "memory"],
        "mode": "sectioned",
        "aliases": [],
        "sections": [
            "8.1. Overall Performance Optimization Strategies",
            "8.2. Maximize Utilization",
            "8.2.1. Application Level",
            "8.2.2. Device Level",
            "8.2.3. Multiprocessor Level",
            "8.2.3.1. Occupancy Calculator",
            "8.3. Maximize Memory Throughput",
            "8.3.1. Data Transfer between Host and Device",
            "8.3.2. Device Memory Accesses",
            "Size and Alignment Requirement",
            "Two-Dimensional Arrays",
            "Local Memory",
            "Shared Memory",
            "Constant Memory",
            "Texture and Surface Memory",
            "8.4. Maximize Instruction Throughput",
            "8.5. Minimize Memory Thrashing",
        ],
    },
    "efficient_gemm": {
        "document_title": "Efficient GEMM in CUDA",
        "tags": ["gemm", "cutlass", "mma", "optimization"],
        "mode": "sectioned",
        "aliases": ["efficient_gemm.txt"],
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
            "Hopper Warp Specialization",
            "Warp-Specialized Persistent Cooperative kernel design",
            "Warp-Specialized Persistent Ping-Pong kernel design",
        ],
    },
    "sgemm_sm80_cu.txt": {
        "document_title": "sgemm_sm80.cu",
        "tags": ["gemm", "code", "cutlass", "mma", "sm80"],
        "mode": "full_file",
        "aliases": [],
    },
}


# ---------------------------------------------------------------------
# Contamination filters
# ---------------------------------------------------------------------

BAD_TEXT_PATTERNS = [
    r"\.\.\.\s*\(continues exactly as provided, unchanged\)",
    r"\.\.\.\s*\(continues.*unchanged\)",
    r"Ready for the next file whenever you are\.",
    r"Below is your content wrapped",
    r"copy-paste-ready corpus block",
    r"same rule as before",
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


def build_alias_map(file_specs: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for canonical, spec in file_specs.items():
        alias_map[canonical] = canonical
        for alias in spec.get("aliases", []):
            alias_map[alias] = canonical
    return alias_map


ALIAS_MAP = build_alias_map(FILE_SPECS)


def canonicalize_filename(name: str) -> str:
    """
    Map actual filenames to their canonical spec key.
    Handles files like cute_layouts(2).txt and explicit aliases.
    """
    if name in ALIAS_MAP:
        return ALIAS_MAP[name]

    cleaned = re.sub(r"\(\d+\)(?=(\.[^.]+)?$)", "", name)
    if cleaned in ALIAS_MAP:
        return ALIAS_MAP[cleaned]

    return cleaned


def is_code_line(line: str) -> bool:
    stripped = line.rstrip("\n")
    s = stripped.strip()

    if not s:
        return False

    code_prefixes = (
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
        "half ",
        "bool ",
        "Tensor ",
        "Layout ",
        "Thr",
        "CUTE_",
        "cuda",
        "cp_async",
        "extern __shared__",
    )
    if s.startswith(code_prefixes):
        return True

    if s.startswith("//") or s.startswith("/*") or s.startswith("*") or s.startswith("*/"):
        return True

    if any(tok in s for tok in ("{", "}", ";", "->", "::", "<<<", ">>>")):
        return True

    if stripped.startswith(("  ", "\t")) and (
        "(" in s or ")" in s or "=" in s or "[" in s or "]" in s or ";" in s
    ):
        return True

    return False


def split_into_code_aware_blocks(text: str) -> List[Tuple[str, bool]]:
    lines = text.splitlines()
    if not lines:
        return []

    blocks: List[Tuple[str, bool]] = []
    current_lines: List[str] = []
    current_kind: bool | None = None

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
    """
    Split oversized text into chunks without splitting code blocks.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    blocks = split_into_code_aware_blocks(text)
    chunks: List[str] = []
    current = ""

    def append_chunk(chunk_text: str) -> None:
        chunk_text = chunk_text.strip()
        if chunk_text and not contains_contamination(chunk_text):
            chunks.append(chunk_text)

    for block_text, is_code in blocks:
        if len(block_text) <= max_chars:
            candidate = f"{current}\n\n{block_text}".strip() if current else block_text
            if len(candidate) <= max_chars:
                current = candidate
            else:
                append_chunk(current)
                current = block_text
            continue

        if current:
            append_chunk(current)
            current = ""

        if is_code:
            append_chunk(block_text)
            continue

        paragraphs = [p.strip() for p in block_text.split("\n\n") if p.strip()]
        para_current = ""

        for p in paragraphs:
            candidate = f"{para_current}\n\n{p}".strip() if para_current else p
            if len(candidate) <= max_chars:
                para_current = candidate
            else:
                if para_current:
                    append_chunk(para_current)
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
            append_chunk(para_current)

    if current:
        append_chunk(current)

    return chunks


def find_header_indices(lines: List[str], headers: List[str]) -> List[Tuple[int, str]]:
    normalized_headers = {normalize_line(h): h for h in headers}
    found: List[Tuple[int, str]] = []

    for idx, line in enumerate(lines):
        norm = normalize_line(line)
        if norm in normalized_headers:
            found.append((idx, normalized_headers[norm]))

    return found


def strip_title_line(lines: List[str]) -> List[str]:
    return [line for line in lines if not line.startswith("TITLE:")]


def read_clean_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if contains_contamination(text):
        raise ValueError(f"Contaminated assistant-placeholder text detected in file: {path.name}")
    return text


# ---------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------

def process_full_file(path: Path, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = read_clean_text(path)
    if not text:
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


def process_sectioned_file(path: Path, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    document_title: str = spec["document_title"]
    base_tags: List[str] = spec["tags"]
    headers: List[str] = spec["sections"]

    text = read_clean_text(path)
    lines = strip_title_line(text.splitlines())

    header_positions = find_header_indices(lines, headers)

    if not header_positions:
        raise ValueError(f"No headers found for file: {path.name}")

    found_headers = {header for _, header in header_positions}
    missing_headers = [h for h in headers if h not in found_headers]
    if missing_headers:
        print(f"[WARN] {path.name}: {len(missing_headers)} missing headers")
        for h in missing_headers:
            print(f"       - {h}")

    deduped_positions: List[Tuple[int, str]] = []
    seen = set()
    for idx, header in header_positions:
        key = (idx, header)
        if key not in seen:
            deduped_positions.append((idx, header))
            seen.add(key)

    output: List[Dict[str, Any]] = []

    for section_idx, (start_idx, section_name) in enumerate(deduped_positions):
        body_start = start_idx + 1
        body_end = deduped_positions[section_idx + 1][0] if section_idx + 1 < len(deduped_positions) else len(lines)

        body_lines = lines[body_start:body_end]
        body = "\n".join(body_lines).strip()

        if not body:
            continue
        if contains_contamination(body):
            print(f"[WARN] Skipping contaminated section in {path.name}: {section_name}")
            continue

        subchunks = split_large_text(body)

        for chunk_idx, chunk in enumerate(subchunks):
            if not chunk.strip():
                continue
            if contains_contamination(chunk):
                print(f"[WARN] Skipping contaminated chunk in {path.name}: {section_name} chunk {chunk_idx}")
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
                    "tags": list(base_tags),
                }
            )

    return output


def process_file(path: Path) -> List[Dict[str, Any]]:
    filename = canonicalize_filename(path.name)

    if filename not in FILE_SPECS:
        raise ValueError(f"No hard-coded spec found for file: {path.name}")

    spec = FILE_SPECS[filename]
    mode = spec.get("mode", "sectioned")

    if mode == "full_file":
        return process_full_file(path, spec)
    if mode == "sectioned":
        return process_sectioned_file(path, spec)

    raise ValueError(f"Unknown mode '{mode}' for file: {path.name}")


def main() -> None:
    all_chunks: List[Dict[str, Any]] = []
    seen_ids = set()

    files = sorted(CORPUS_DIR.glob("*"))

    for path in files:
        if not path.is_file():
            continue

        canonical = canonicalize_filename(path.name)
        if canonical not in FILE_SPECS:
            print(f"[INFO] Skipping unrecognized file: {path.name}")
            continue

        try:
            chunks = process_file(path)
        except Exception as e:
            print(f"[ERROR] {path.name}: {e}")
            continue

        for chunk in chunks:
            if chunk["id"] in seen_ids:
                raise ValueError(f"Duplicate chunk id detected: {chunk['id']}")
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