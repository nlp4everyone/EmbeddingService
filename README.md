# EmbeddingService

A local embedding-model serving environment built on [vLLM](https://github.com/vllm-project/vllm), exposing an OpenAI-compatible `/v1/embeddings` API and optimized for Nvidia GPU acceleration.

## Prerequisites

### Hardware
- Nvidia GPU with Compute Capability 7.0 or higher
- At least 8GB of GPU VRAM
- 16GB+ system RAM

### Software
- Linux-based operating system (Ubuntu 20.04/22.04 recommended)
- Nvidia Driver 535.54.03 or later
- Docker 20.10.0 or later
- Docker Compose v2.0.0 or later
- Nvidia Container Toolkit

## Configuration

Copy `.env.sample` to `.env` and adjust as needed:

```bash
# Port config
VLLM_EMBEDDING_PORT=8100
# Model config
EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_GPU_MEM_UTIL=0.6
SERVING_API_KEY=token
```

## Quick Start

1. **Start the service**
   ```bash
   make up
   ```

2. **Verify it's healthy**
   ```bash
   make status
   ```

3. **Send a test embedding request**
   ```bash
   make test
   ```

Other Makefile targets: `make down`, `make restart`, `make logs`, `make ps`, `make pull`, `make clean`.

## Serving Multiple Models

Each model is defined by its own compose file under `docker/`. `compose_serving.yml` currently serves `Qwen/Qwen3-Embedding-0.6B`.

Select which one a command targets by passing its name as a second word:

```bash
make up qwen3      # docker/compose_serving.yml
make up bge-m3      # docker/bge_compose_serving.yml
```

To add a new model, create its compose file under `docker/` and register it in the `Makefile`:

```make
<name>_FILE := <compose-file>.yml
MODELS      := qwen3 bge-m3 <name>
```

## Fetching Embeddings from Python

`scripts/qwen3_embedding_sample.py` shows how to call the running service using both the `openai` client and raw `requests`:

```bash
pip install openai requests python-dotenv
python scripts/qwen3_embedding_sample.py "Hello world" "Another sentence"
```

## Troubleshooting

### Nvidia Driver Issues
If you get an `NVIDIA-SMI has failed` error, verify your drivers are installed correctly:
```bash
nvidia-smi
```
If the command is not found, reinstall Nvidia drivers and reboot your system.

### Container Fails to See the GPU
Ensure the Nvidia Container Toolkit is installed and Docker's default runtime is configured for it, then restart Docker:
```bash
sudo systemctl restart docker
```