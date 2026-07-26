ENV_FILE := .env
include $(ENV_FILE)
export

# Map model name -> compose file. Add a new line here for each new model.
dense_FILE  := dense_compose_serving.yml
sparse_FILE := sparse_compose_serving.yml
hybrid_FILE := hybrid_compose_serving.yml
MODELS      := dense sparse hybrid
DEFAULT_FILE := dense_compose_serving.yml

# Second word on the command line selects the model, e.g. `make up sparse`.
MODEL        := $(word 2,$(MAKECMDGOALS))
COMPOSE_FILE := docker/$(if $(MODEL),$($(MODEL)_FILE),$(DEFAULT_FILE))
COMPOSE      := docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE)

.PHONY: up down restart logs ps pull status test clean check-model $(MODELS)

# down/logs/ps/pull act on whichever compose file is currently running, so a
# missing model would silently fall back to DEFAULT_FILE and look like a bug
# (e.g. "make logs" showing nothing while "make up hybrid" is running).
check-model:
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: specify a model, e.g. 'make $(firstword $(MAKECMDGOALS)) hybrid' (options: $(MODELS))"; \
		exit 1; \
	fi

up: ## make up [dense|sparse|hybrid] — start the selected service (defaults to dense)
	$(COMPOSE) up -d

down: check-model ## make down <dense|sparse|hybrid> — stop the selected service
	$(COMPOSE) down

restart: down up ## make restart <dense|sparse|hybrid>

logs: check-model ## make logs <dense|sparse|hybrid>
	$(COMPOSE) logs -f

ps: check-model ## make ps <dense|sparse|hybrid>
	$(COMPOSE) ps

pull: check-model ## make pull <dense|sparse|hybrid>
	$(COMPOSE) pull

status: ## Check the dense embedding endpoint health
	curl -sf http://localhost:$(TEI_DENSE_EMBEDDING_PORT)/health && echo "OK"

test: ## Send a sample embedding request to the dense endpoint
	curl -s http://localhost:$(TEI_DENSE_EMBEDDING_PORT)/v1/embeddings \
		-H "Authorization: Bearer $(SERVING_API_KEY)" \
		-H "Content-Type: application/json" \
		-d '{"model": "$(DENSE_MODEL_NAME)", "input": "Hello world"}'

clean: down ## Stop the service and remove the model cache volume
	docker volume rm tei-cache

# Swallow the model name so make doesn't treat it as an unknown target.
$(MODELS):
	@: