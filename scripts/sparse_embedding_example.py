#!/usr/bin/env python3
"""Fetch sparse (SPLADE) embeddings from the locally served Text Embeddings Inference endpoint."""

# Typing hints
from typing import List, Dict
# Other components
import httpx, asyncio
from dotenv import load_dotenv
import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

PORT = os.environ.get("TEI_SPARSE_EMBEDDING_PORT", "8101")
API_KEY = os.environ.get("SERVING_API_KEY", "token")
MODEL = os.environ.get("SPARSE_MODEL_NAME", "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1")
BASE_URL = f"http://localhost:{PORT}"

async def get_sparse_embeddings(sentences: List[str],
                                timeout: float = 3.0) -> List[Dict[int, float]]:
    """Generate sparse embeddings (lexical weights) using the SPLADE model.

    Calls the TEI `/embed_sparse` endpoint, which returns each input's sparse
    representation directly as a list of {index, value} pairs over the
    model's vocabulary, so no separate tokenization step is needed.

    Args:
        sentences: List of input sentences to embed.
        timeout: Request timeout in seconds. Defaults to 3.0 seconds.

    Returns:
        List of sparse embeddings, one per sentence, mapping token id to weight.
    """
    url = f"{BASE_URL}/embed_sparse"

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            url,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"inputs": sentences},
        )

    response.raise_for_status()
    return [{item["index"]: item["value"] for item in embedding} for embedding in response.json()]

def main():
    # Demo entrypoint: fetch and print sparse embeddings for a couple of sample sentences
    sentences = ["Hello world", "Another sentence"]
    print(f"Requesting sparse embeddings for {len(sentences)} sentence(s) from {BASE_URL}...")
    embeddings = asyncio.run(get_sparse_embeddings(sentences))
    for text, embedding in zip(sentences, embeddings):
        print(f"[requests] Content:{text!r}. Sparse embedding: {embedding}")

if __name__ == "__main__":
    main()