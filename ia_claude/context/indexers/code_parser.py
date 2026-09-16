from pathlib import Path
from dataclasses import dataclass

from tree_sitter_languages import get_parser

from ia_claude.observability.logger import get_logger


logger = get_logger(__name__)


# Maps file extension -> Tree-sitter language name.
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".sh": "bash",
}


# Text/config files don't have a useful AST for our purposes,
# so we chunk them by line count instead.
TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
}


ALL_EXTENSIONS = set(EXTENSION_TO_LANGUAGE.keys()) | TEXT_EXTENSIONS


# Sliding-window settings for text files and AST fallback.
CHUNK_SIZE = 50
CHUNK_OVERLAP = 10


# Tree-sitter node types that represent named/indexable blocks.
BLOCK_NODE_TYPES = {
    "function_definition",
    "function_declaration",
    "method_definition",
    "arrow_function",
    "class_definition",
    "class_declaration",
    "method_declaration",
    "constructor_declaration",
    "interface_declaration",
    "function_item",       # Rust
    "func_declaration",    # Go
}


@dataclass
class ParsedChunk:
    name: str
    type: str
    content: str
    source: str
    start_line: int
    end_line: int


def parse_file(filepath: str) -> list[ParsedChunk]:
    """
    Entry point.

    Routes code files through Tree-sitter AST parsing
    and text/config files through sliding-window chunking.
    """

    ext = Path(filepath).suffix.lower()

    if ext in TEXT_EXTENSIONS:
        lines = (
            Path(filepath)
            .read_text(
                encoding="utf-8",
                errors="ignore",
            )
            .splitlines()
        )

        return _sliding_window(lines, filepath)

    language_name = EXTENSION_TO_LANGUAGE.get(ext)

    if not language_name:
        raise ValueError(
            f"Unsupported file type: {ext}"
        )

    source = Path(filepath).read_text(
        encoding="utf-8",
        errors="ignore",
    )

    return _parse_with_treesitter(
        source,
        filepath,
        language_name,
    )


def _parse_with_treesitter(
    source: str,
    filepath: str,
    language_name: str,
) -> list[ParsedChunk]:
    """
    Parse source code with Tree-sitter and extract
    meaningful code blocks such as functions and classes.
    """

    logger.info(
        f"Parsing {language_name} file: {filepath}"
    )

    parser = get_parser(language_name)

    tree = parser.parse(
        source.encode("utf-8")
    )

    lines = source.splitlines()

    chunks: list[ParsedChunk] = []

    _walk(
        tree.root_node,
        source,
        filepath,
        chunks,
        depth=0,
    )

    # If no AST blocks were found, fall back
    # to normal line-based chunks.
    if not chunks:
        logger.warning(
            f"No AST blocks found in {filepath}, "
            "falling back to sliding window"
        )

        return _sliding_window(
            lines,
            filepath,
        )

    logger.info(
        f"Parsed {len(chunks)} chunks from {filepath}"
    )

    return chunks


def _walk(
    node,
    source: str,
    filepath: str,
    chunks: list[ParsedChunk],
    depth: int,
):
    """
    Recursively walk the AST.

    When we find a named block, record it as a chunk
    and stop descending into it.
    """

    if node.type in BLOCK_NODE_TYPES:

        name = _extract_name(
            node,
            source,
        )

        # Tree-sitter byte offsets refer to the UTF-8
        # encoded source, so slice encoded bytes first.
        source_bytes = source.encode("utf-8")

        content = source_bytes[
            node.start_byte:node.end_byte
        ].decode(
            "utf-8",
            errors="ignore",
        )

        chunk_type = (
            "class"
            if "class" in node.type
            else "function"
        )

        chunks.append(
            ParsedChunk(
                name=name,
                type=chunk_type,
                content=content,
                source=str(
                    Path(filepath).resolve()
                ),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            )
        )

        logger.debug(
            f"Found {chunk_type} '{name}' "
            f"(lines "
            f"{node.start_point[0] + 1}-"
            f"{node.end_point[0] + 1})"
        )

        # Don't separately index nested
        # functions/classes.
        return

    for child in node.children:
        _walk(
            child,
            source,
            filepath,
            chunks,
            depth + 1,
        )


def _extract_name(
    node,
    source: str,
) -> str:
    """
    Find the identifier/name of an AST node.

    Tree-sitter positions are byte offsets, so we
    extract the name from UTF-8 encoded source bytes.
    """

    source_bytes = source.encode("utf-8")

    for child in node.children:
        if child.type in (
            "identifier",
            "name",
            "property_identifier",
        ):
            return source_bytes[
                child.start_byte:child.end_byte
            ].decode(
                "utf-8",
                errors="ignore",
            )

    return node.type


def _sliding_window(
    lines: list[str],
    filepath: str,
) -> list[ParsedChunk]:
    """
    Split text into overlapping fixed-size chunks.

    Used for:
    - Markdown
    - YAML
    - JSON
    - TOML
    - text files
    - code files where AST parsing found nothing
    """

    if not lines:
        raise ValueError(
            f"Empty file: {filepath}"
        )

    chunks: list[ParsedChunk] = []

    step = CHUNK_SIZE - CHUNK_OVERLAP

    for i, start in enumerate(
        range(0, len(lines), step)
    ):
        end = min(
            start + CHUNK_SIZE,
            len(lines),
        )

        text = "\n".join(
            lines[start:end]
        ).strip()

        if text:
            chunks.append(
                ParsedChunk(
                    name=f"chunk_{i}",
                    type="block",
                    content=text,
                    source=str(
                        Path(filepath).resolve()
                    ),
                    start_line=start + 1,
                    end_line=end,
                )
            )

        if end == len(lines):
            break

    logger.info(
        f"Parsed {len(chunks)} chunks from {filepath}"
    )

    return chunks


def get_source_files(
    repo_path: str,
    skip_dirs: list[str] | None = None,
) -> list[str]:
    """
    Recursively find all indexable source files
    under repo_path.
    """

    repo_path = str(
        Path(repo_path).resolve()
    )

    skip = set(
        skip_dirs
        or [
            ".venv",
            "venv",
            "__pycache__",
            ".git",
            "node_modules",
            "dist",
            "build",
            ".ia_claude_history_exports",
        ]
    )

    files = [
        str(path.resolve())
        for path in Path(repo_path).rglob("*")
        if path.is_file()
        and path.suffix.lower() in ALL_EXTENSIONS
        and not any(
            part in skip
            for part in path.parts
        )
    ]

    logger.info(
        f"Found {len(files)} source files "
        f"in {repo_path}"
    )

    return files
