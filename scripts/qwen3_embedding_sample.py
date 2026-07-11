#!/usr/bin/env python3
"""Fetch text embeddings from the locally served Qwen3 embedding model (vLLM)."""

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

PORT = os.environ.get("VLLM_EMBEDDING_PORT", "8100")
API_KEY = os.environ.get("SERVING_API_KEY", "token")
MODEL = os.environ.get("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
BASE_URL = f"http://localhost:{PORT}/v1"


def get_embedding_openai(texts: list[str]) -> list[list[float]]:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    response = client.embeddings.create(model=MODEL, input=texts)
    return [item.embedding for item in response.data]


def get_embedding_requests(texts: list[str]) -> list[list[float]]:
    response = requests.post(
        f"{BASE_URL}/embeddings",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": MODEL, "input": texts},
    )
    response.raise_for_status()
    return [item["embedding"] for item in response.json()["data"]]


if __name__ == "__main__":
    texts = sys.argv[1:] or ["Hello world"]

    for text, embedding in zip(texts, get_embedding_openai(texts)):
        print(f"[openai]   {text!r}: dim={len(embedding)} {embedding[:8]}...")

    for text, embedding in zip(texts, get_embedding_requests(texts)):
        print(f"[requests] {text!r}: dim={len(embedding)} {embedding[:8]}...")