# 🧠 EmbeddingService

A local embedding-model serving environment built on [vLLM](https://github.com/vllm-project/vllm), exposing an OpenAI-compatible `/v1/embeddings` API and optimized for Nvidia GPU acceleration.

## 📋 Prerequisites

### 🖥️ Hardware
- Nvidia GPU with Compute Capability 7.0 or higher
- At least 8GB of GPU VRAM
- 16GB+ system RAM

### 🛠️ Software
- Linux-based operating system (Ubuntu 20.04/22.04 recommended)
- Nvidia Driver 535.54.03 or later
- Docker 20.10.0 or later
- Docker Compose v2.0.0 or later
- Nvidia Container Toolkit

## ⚙️ Configuration

Copy `.env.sample` to `.env` and adjust as needed:

```bash
cp .env.sample .env
```

```bash
# Port config
SERVING_API_KEY=token

# Model config (dense = Qwen3, sparse = BGE-M3)
VLLM_DENSE_EMBEDDING_PORT=8100
VLLM_SPARSE_EMBEDDING_PORT=8101
DENSE_MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
SPARSE_MODEL_NAME=BAAI/bge-m3
DENSE_GPU_MEM_UTIL=0.6
SPARSE_GPU_MEM_UTIL=0.3
```

## 🚀 Quick Start

1. **Start the service** ▶️
   ```bash
   make up
   ```

2. **Verify it's healthy** ✅
   ```bash
   make status
   ```
   ```text
   OK
   ```

3. **Send a test embedding request** 🔎
   ```bash
   make test
   ```
   ```json
   {"id":"embd-...","object":"list","data":[{"index":0,"embedding":[0.0123,-0.0456,...]}],"model":"Qwen/Qwen3-Embedding-0.6B"}
   ```

Other Makefile targets: `make down`, `make restart`, `make logs`, `make ps`, `make pull`, `make clean`.

## 🔀 Serving Multiple Models

Each model is defined by its own compose file under `docker/`. `dense_compose_serving.yml` currently serves `Qwen/Qwen3-Embedding-0.6B`.

Select which one a command targets by passing its name as a second word:

```bash
make up dense       # 🟣 docker/dense_compose_serving.yml  — dense embeddings only
make up sparse      # 🟢 docker/sparse_compose_serving.yml — sparse embeddings only
make up hybrid      # 🟡 docker/hybrid_compose_serving.yml — dense + sparse together
```

`hybrid` runs both models at once as separate containers (`vllm_bge`, `vllm_qwen3`), each on its own port (`VLLM_SPARSE_EMBEDDING_PORT`/`VLLM_DENSE_EMBEDDING_PORT`, default `8100`/`8101`) and sharing the GPU via `SPARSE_GPU_MEM_UTIL`/`DENSE_GPU_MEM_UTIL` (default `0.3`/`0.6`). Adjust the memory utilization values in `.env` so they fit within your GPU's VRAM.

```bash
make up hybrid
make ps hybrid
```
```text
NAME                    SERVICE      STATUS          PORTS
vllm_dense_embedding    vllm_qwen3   Up 14 seconds   0.0.0.0:8101->8000/tcp
vllm_sparse_embedding   vllm_bge     Up 2 minutes    0.0.0.0:8100->8000/tcp
```

⚠️ `down`, `logs`, `ps`, and `pull` need the model name too once a non-default model is running (e.g. `make logs hybrid`) — without it they'd silently target the default `dense` compose file and appear to hang or show nothing, so they now fail fast with a reminder instead:

```bash
$ make logs
Error: specify a model, e.g. 'make logs hybrid' (options: dense sparse hybrid)
```

To add a new model, create its compose file under `docker/` and register it in the `Makefile`:

```make
<name>_FILE := <compose-file>.yml
MODELS      := dense sparse <name>
```

## 🐍 Fetching Embeddings from Python

```bash
pip install openai requests httpx python-dotenv
```

| Script | Model(s) | Embedding type | Example |
| --- | --- | --- | --- |
| 🟣 `scripts/dense_embedding_example.py` | Qwen3 | Dense | `python scripts/dense_embedding_example.py "Hello world" "Another sentence"` |
| 🟢 `scripts/sparse_embedding_example.py` | BGE-M3 | Sparse | `python scripts/sparse_embedding_example.py` |
| 🟡 `scripts/hybrid_embedding_example.py` | BGE-M3 + Qwen3 | Dense + Sparse | `python scripts/hybrid_embedding_example.py` |

`dense_embedding_example.py` calls the running service using both the `openai` client and raw `requests`, and accepts sentences as CLI args:

```bash
python scripts/dense_embedding_example.py "Hello world" "Another sentence"
```
```text
[openai]   'Hello world': dim=1024 [0.012, -0.034, 0.056, ...]...
[requests] 'Hello world': dim=1024 [0.012, -0.034, 0.056, ...]...
```

`sparse_embedding_example.py` tokenizes each sentence, then pools it into a sparse `{token_id: weight}` map:

```bash
python scripts/sparse_embedding_example.py
```
```text
[requests] Content:'Hello world'. Sparse embedding: {8994: 0.213, 1362: 0.187, ...}
```

`hybrid_embedding_example.py` requires `make up hybrid` and fetches both embedding types concurrently:

```bash
python scripts/hybrid_embedding_example.py
```
```text
[dense]  'Hello world': dim=1024 [0.012, -0.034, ...]...
[sparse] 'Hello world': {8994: 0.213, 1362: 0.187, ...}
```

## 🩺 Troubleshooting

### 🔌 Nvidia Driver Issues
If you get an `NVIDIA-SMI has failed` error, verify your drivers are installed correctly:
```bash
nvidia-smi
```
If the command is not found, reinstall Nvidia drivers and reboot your system.

### 🚫 Container Fails to See the GPU
Ensure the Nvidia Container Toolkit is installed and Docker's default runtime is configured for it, then restart Docker:
```bash
sudo systemctl restart docker
```