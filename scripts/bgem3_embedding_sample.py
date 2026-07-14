# Typing hints
from typing import List, Dict, Any
# Other components
import httpx, asyncio
from dotenv import load_dotenv
import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

PORT = os.environ.get("VLLM_EMBEDDING_PORT", "8100")
API_KEY = os.environ.get("SERVING_API_KEY", "token")
MODEL = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
BASE_URL = f"http://localhost:{PORT}/v1"

async def tokenize(sentences: List[str],
                   timeout :float = 3.0) -> List[List[int]]:
    """Tokenize a list of sentences into token IDs using the embedding service.

    This function sends each sentence to the embedding service's tokenization
    endpoint and returns the corresponding token IDs. The requests are made
    concurrently for improved performance.

    Args:
        sentences: List of input sentences to tokenize. Each sentence should
            be a string that can be tokenized by the embedding model.
        timeout: Request timeout in seconds. Defaults to 3.0 seconds.

    Returns:
        List of token ID lists, one for each input sentence
    """
    # Construct the tokenization API URL endpoint
    url = f"http://localhost:{PORT}/tokenize"

    # Create async client and prepare concurrent requests for all sentences
    # Each sentence is tokenized independently to ensure proper tokenization
    async with httpx.AsyncClient(timeout = timeout) as client:
        tasks = [
            client.post(url, json = {"model": MODEL,
                                     "prompt": sentence})
            for sentence in sentences
        ]

        # Execute all tokenization requests concurrently for better performance
        responses: List[httpx.Response] = await asyncio.gather(*tasks)

    # Extract token IDs from each response and return the token lists
    return [response.json()["tokens"] for response in responses]

async def get_sparse_embeddings(sentences: List[str],
                                timeout :float = 3.0) -> Dict[str, Any]:
    """Generate sparse embeddings (lexical weights) using BGE-M3 token classification.

    This function produces sparse embeddings that represent lexical weights for
    tokens in each sentence. It uses a two-step process: first tokenization,
    then token classification to generate weight mappings for each token.

    Args:
        sentences: List of input sentences to embed. Each sentence should be
            a string that can be processed by the embedding model.
        timeout: Request timeout in seconds. Defaults to 3.0 seconds.

    Returns:
        List[SparseEmbedding]: List of sparse embeddings, where each embedding
            contains a dictionary mapping token IDs to their corresponding weights.
            The sparse representation is efficient for text with many zero weights.

    Raises:
        httpx.TimeoutException: If any request exceeds the specified timeout.
        httpx.HTTPStatusError: If the embedding service returns an error status.
        KeyError: If the response format is unexpected.

    """
    # Step 1: Tokenize all sentences to get token IDs
    all_tokens = await tokenize(sentences, timeout = timeout)

    # Step 2: Call pooling API with token_classify task for sparse embeddings
    url = f"http://localhost:{PORT}/pooling"
    payload = {
        "model": MODEL,
        #"task": "token_classify",
        "input": sentences
    }

    async with httpx.AsyncClient(timeout = timeout) as client:
        result: httpx.Response = await client.post(url, json=payload)

    # Extract embedding data from the response
    all_embeddings = [data["data"] for data in result.json()["data"]]

    # Step 3: Build sparse dictionary mapping {token_id: weight} for each sentence
    ret = []

    for sent_tokens, sent_emb in zip(all_tokens, all_embeddings):
        token_embs = {}

        # Remove BOS (Beginning of Sequence) token if present (token ID 0)
        if sent_tokens and sent_tokens[0] == 0:
            sent_tokens = sent_tokens[1:]

        # Map each token to its weight, keeping the maximum weight for duplicate tokens
        for token, val in zip(sent_tokens, sent_emb):
            token_embs[token] = max(val, token_embs.get(token, 0.0))
        ret.append(token_embs)

    return {"data": ret}

def main():
    # Demo entrypoint: fetch and print sparse embeddings for a couple of sample sentences
    sentences = ["Hello world", "Another sentence"]
    print(f"Requesting sparse embeddings for {len(sentences)} sentence(s) from {BASE_URL}...")
    response = asyncio.run(get_sparse_embeddings(sentences))
    for text, embedding in zip(sentences, response.get("data")):
        print(f"[requests] Content:{text!r}. Sparse embedding: {embedding}")

if __name__ == "__main__":
    main()
