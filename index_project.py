# index_project.py

"""
index_project.py — Poshak-AI
────────────────────────────
Indexes one or more project directories into ChromaDB.
Supports multi-repo PROJECT_GROUPS — all repos indexed into one collection.

Usage:
    python index_project.py                    # indexes ACTIVE_PROJECT
    python index_project.py --list             # lists all configured projects/groups
"""

import os
import sys
import time
import argparse
from pathlib import Path

# ── Windows UTF-8 fix ─────────────────────────────────────────────────────────
# Windows console defaults to cp1252 which can't encode box-drawing chars (─, ✓, etc.)
# Reconfigure stdout/stderr to UTF-8 at the very top before any print() calls.
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import (
    PROJECTS,
    PROJECT_GROUPS,
    ACTIVE_PROJECT,
    ALL_EXTENSIONS,
    EXCLUDE_DIRS,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    DB_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    get_active_project_paths,
    validate_config,
)


def info(msg: str):  print(f"  [+] {msg}")
def ok(msg: str):    print(f"  [✓] {msg}")
def warn(msg: str):  print(f"  [!] {msg}")
def err(msg: str):   print(f"  [✗] {msg}")

SEPARATOR = "─" * 56


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in path.parts)


def load_files_from_path(project_path: str, sub_project_label: str = "") -> list:
    """Load all matching files from a single directory."""
    from langchain_community.document_loaders import TextLoader

    project_path = Path(project_path).resolve()
    if not project_path.exists():
        err(f"Path not found: {project_path}")
        return []

    matched_files = set()
    for pattern in ALL_EXTENSIONS:
        for file in project_path.glob(pattern):
            if not is_excluded(file.relative_to(project_path)):
                matched_files.add(file)

    docs = []
    skipped = 0
    extensions_seen = {}

    for file in sorted(matched_files):
        ext = file.suffix.lower()
        for encoding in ("utf-8", "latin-1"):
            try:
                loader = TextLoader(str(file), encoding=encoding)
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source"]      = str(file.relative_to(project_path))
                    doc.metadata["extension"]   = ext
                    doc.metadata["filename"]    = file.name
                    doc.metadata["sub_project"] = sub_project_label
                docs.extend(loaded)
                extensions_seen[ext] = extensions_seen.get(ext, 0) + 1
                break
            except UnicodeDecodeError:
                continue
            except Exception:
                skipped += 1
                break

    if skipped:
        warn(f"Skipped {skipped} files in '{sub_project_label or project_path}' (unreadable)")

    return docs, extensions_seen


def split_documents(docs: list) -> list:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    separators = [
        "\npublic class ", "\nprivate class ", "\nprotected class ",
        "\npublic interface ", "\npublic enum ",
        "\n    public ", "\n    private ", "\n    protected ",
        "\nexport default ", "\nexport const ", "\nexport function ",
        "\nexport class ", "\nconst ", "\nfunction ", "\nclass ",
        "\n\n\n", "\n\n", "\n", " ", "",
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    avg = sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)
    info(f"Split into {len(chunks)} chunks (avg {avg} chars each)")
    return chunks


def load_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
    ok("Embedding model loaded")
    return embeddings


def build_vector_store(chunks: list, embeddings):
    import chromadb
    from langchain_chroma import Chroma

    client = chromadb.PersistentClient(path=DB_PATH)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        warn(f"Replacing existing collection '{COLLECTION_NAME}'")
        client.delete_collection(COLLECTION_NAME)

    info("Building vector database...")

    BATCH = 200
    total = len(chunks)
    db    = None

    for i in range(0, total, BATCH):
        batch = chunks[i: i + BATCH]
        pct   = min(i + BATCH, total)
        print(f"\r       {pct}/{total} chunks indexed...", end="", flush=True)
        if db is None:
            db = Chroma.from_documents(
                batch, embeddings,
                persist_directory=DB_PATH,
                collection_name=COLLECTION_NAME,
            )
        else:
            db.add_documents(batch)

    print()
    return db


def index_workspace(workspace_name: str):
    # Resolve paths — works for both PROJECTS and PROJECT_GROUPS
    if workspace_name in PROJECT_GROUPS:
        sub_projects = PROJECT_GROUPS[workspace_name]
    elif workspace_name in PROJECTS:
        sub_projects = {"main": PROJECTS[workspace_name]}
    else:
        err(f"'{workspace_name}' not found in PROJECTS or PROJECT_GROUPS")
        raise SystemExit(1)

    print()
    print(SEPARATOR)
    print(f"  Poshak-AI — Indexing workspace: {workspace_name}")
    print(SEPARATOR)
    for label, path in sub_projects.items():
        info(f"  {label}: {path}")
    info(f"DB: {DB_PATH}/{COLLECTION_NAME}")
    print()

    start = time.time()
    all_docs = []
    all_extensions: dict = {}

    # Step 1: Load all sub-projects
    info(f"Step 1/4 — Loading source files ({len(sub_projects)} sub-project(s))")
    for label, path in sub_projects.items():
        print(f"\n  -- Sub-project: {label} --")
        docs, exts = load_files_from_path(path, sub_project_label=label)
        all_docs.extend(docs)
        for ext, count in exts.items():
            all_extensions[ext] = all_extensions.get(ext, 0) + count
        ok(f"  {len(docs)} files loaded from '{label}'")

    if not all_docs:
        err("No files loaded. Check your paths in config.py")
        raise SystemExit(1)

    print()
    for ext, count in sorted(all_extensions.items(), key=lambda x: -x[1]):
        print(f"       {ext or 'no-ext':14s}  {count} files")
    print()
    ok(f"Total: {len(all_docs)} files across {len(sub_projects)} sub-project(s)")

    # Step 2: Split
    print()
    info("Step 2/4 — Splitting into chunks")
    chunks = split_documents(all_docs)

    # Step 3: Embeddings
    print()
    info("Step 3/4 — Loading embedding model")
    embeddings = load_embeddings()

    # Step 4: Vector store
    print()
    info("Step 4/4 — Building ChromaDB vector store")
    build_vector_store(chunks, embeddings)

    elapsed = round(time.time() - start, 1)
    print()
    print(SEPARATOR)
    ok(f"Indexing complete in {elapsed}s")
    ok(f"{len(chunks)} chunks stored — all sub-projects searchable together")
    ok("Run: python agent.py  or  python ui.py")
    print(SEPARATOR)
    print()


def list_all():
    print()
    print("  Single projects (PROJECTS in config.py):")
    for name, path in PROJECTS.items():
        exists = os.path.isdir(path)
        active = " <- ACTIVE" if name == ACTIVE_PROJECT else ""
        print(f"    [{'OK' if exists else 'XX'}] {name}{active}  ->  {path}")

    print()
    print("  Project groups (PROJECT_GROUPS in config.py):")
    for name, subs in PROJECT_GROUPS.items():
        active = " <- ACTIVE" if name == ACTIVE_PROJECT else ""
        print(f"    {name}{active}")
        for label, path in subs.items():
            exists = os.path.isdir(path)
            print(f"      [{'OK' if exists else 'XX'}] {label}  ->  {path}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index a project/group into Poshak-AI")
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        list_all()
        sys.exit(0)

    validate_config()
    target = args.project or ACTIVE_PROJECT
    index_workspace(target)