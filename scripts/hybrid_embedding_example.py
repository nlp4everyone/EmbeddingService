#!/usr/bin/env python3
"""Fetch both dense (Qwen3) and sparse (SPLADE) embeddings from the hybrid serving stack."""

# Typing hints
from typing import List, Dict, Any
# Other components
import httpx, asyncio
from dotenv import load_dotenv
import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

API_KEY = os.environ.get("SERVING_API_KEY", "token")

SPARSE_PORT = os.environ.get("TEI_SPARSE_EMBEDDING_PORT", "8101")
SPARSE_MODEL = os.environ.get("SPARSE_MODEL_NAME", "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1")

DENSE_PORT = os.environ.get("TEI_DENSE_EMBEDDING_PORT", "8100")
DENSE_MODEL = os.environ.get("DENSE_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")


async def get_dense_embeddings(sentences: List[str],
                               timeout: float = 3.0) -> List[List[float]]:
    """Fetch dense embeddings for each sentence from the Qwen3 service."""
    url = f"http://localhost:{DENSE_PORT}/v1/embeddings"

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            url,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": DENSE_MODEL, "input": sentences},
        )

    return [item["embedding"] for item in response.json()["data"]]


async def get_sparse_embeddings(sentences: List[str],
                                timeout: float = 3.0) -> List[Dict[int, float]]:
    """Fetch sparse embeddings (lexical weights) for each sentence from the SPLADE service."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"http://localhost:{SPARSE_PORT}/embed_sparse",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"inputs": sentences},
        )

    return [{item["index"]: item["value"] for item in embedding} for embedding in response.json()]


async def get_hybrid_embeddings(sentences: List[str],
                                timeout: float = 3.0) -> Dict[str, Any]:
    """Fetch dense and sparse embeddings concurrently from the two hybrid services."""
    dense, sparse = await asyncio.gather(
        get_dense_embeddings(sentences, timeout=timeout),
        get_sparse_embeddings(sentences, timeout=timeout),
    )
    return {"dense": dense, "sparse": sparse}


def main():
    sentences = ["Hello world", "Another sentence"]
    print(f"Requesting dense (:{DENSE_PORT}) and sparse (:{SPARSE_PORT}) embeddings "
          f"for {len(sentences)} sentence(s)...")
    result = asyncio.run(get_hybrid_embeddings(sentences))

    for text, dense, sparse in zip(sentences, result["dense"], result["sparse"]):
        print(f"[dense]  {text}: dim={len(dense)} {dense[:8]}...")
        print(f"[sparse] {text}: {sparse}")


if __name__ == "__main__":
    main()