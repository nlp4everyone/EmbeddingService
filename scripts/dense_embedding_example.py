#!/usr/bin/env python3
"""Fetch text embeddings from the locally served Qwen3 embedding model (Text Embeddings Inference)."""

# Typing hints
from typing import List
# Other components
import httpx, asyncio
from dotenv import load_dotenv
import os
import sys
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

PORT = os.environ.get("TEI_DENSE_EMBEDDING_PORT", "8100")
API_KEY = os.environ.get("SERVING_API_KEY", "token")
BASE_URL = f"http://localhost:{PORT}"


async def get_dense_embeddings(texts: List[str],
                               timeout: float = 3.0) -> List[List[float]]:
    """Fetch dense embeddings for each text from the Qwen3 embedding service."""
    url = f"{BASE_URL}/embed"

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            url,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"inputs": texts},
        )

    response.raise_for_status()
    return response.json()


def main():
    texts = sys.argv[1:] or ["Hello world"]
    print(f"Requesting dense embeddings for {len(texts)} text(s) from {BASE_URL}...")
    embeddings = asyncio.run(get_dense_embeddings(texts))
    for text, embedding in zip(texts, embeddings):
        print(f"[httpx] {text!r}: dim={len(embedding)} {embedding[:8]}...")


if __name__ == "__main__":
    main()