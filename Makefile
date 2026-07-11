ENV_FILE := .env
include $(ENV_FILE)
export

# Map model name -> compose file. Add a new line here for each new model.
qwen3_FILE  := compose_serving.yml
bge-m3_FILE := bge_compose_serving.yml
MODELS      := qwen3 bge-m3
DEFAULT_FILE := compose_serving.yml

# Second word on the command line selects the model, e.g. `make up qwen3`.
MODEL        := $(word 2,$(MAKECMDGOALS))
COMPOSE_FILE := docker/$(if $(MODEL),$($(MODEL)_FILE),$(DEFAULT_FILE))
COMPOSE      := docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE)

.PHONY: up down restart logs ps pull status test clean $(MODELS)

up: ## make up [qwen3|bge-m3] — start the selected service
	$(COMPOSE) up -d

down: ## make down [qwen3|bge-m3] — stop the selected service
	$(COMPOSE) down

restart: down up ## make restart [qwen3|bge-m3]

logs: ## make logs [qwen3|bge-m3]
	$(COMPOSE) logs -f

ps: ## make ps [qwen3|bge-m3]
	$(COMPOSE) ps

pull: ## make pull [qwen3|bge-m3]
	$(COMPOSE) pull

status: ## Check embedding endpoint health
	curl -sf http://localhost:$(VLLM_EMBEDDING_PORT)/health && echo "OK"

test: ## Send a sample embedding request
	curl -s http://localhost:$(VLLM_EMBEDDING_PORT)/v1/embeddings \
		-H "Authorization: Bearer $(SERVING_API_KEY)" \
		-H "Content-Type: application/json" \
		-d '{"model": "$(EMBEDDING_MODEL_NAME)", "input": "Hello world"}'

clean: down ## Stop the service and remove the model cache volume
	docker volume rm vllm-cache

# Swallow the model name so make doesn't treat it as an unknown target.
$(MODELS):
	@: