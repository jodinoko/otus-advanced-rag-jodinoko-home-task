from __future__ import annotations

import argparse

from corporate_rag.indexing import build_index


def index_main() -> None:
    parser = argparse.ArgumentParser(description="Index corporate PDF and HTML sources in Qdrant.")
    parser.add_argument("--reset", action="store_true", help="Replace the existing collection.")
    args = parser.parse_args()
    _, nodes = build_index(reset=args.reset)
    print(f"Indexed {len(nodes)} chunks in Qdrant.")
