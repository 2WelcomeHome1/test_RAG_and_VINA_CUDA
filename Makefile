# Server (H100) lifecycle: build the image, download the models once, start the stack.
# Model names come from .env (as docker compose reads them), else the defaults below, and are
# exported to docker compose, so the pull and the containers agree.
# Override any of them per call: make start LLM_MODEL=qwen3:32b

-include .env

COMPOSE := docker compose -f docker-compose.yml -f docker-compose.gpu.yml

export RETRIEVER_MODEL ?= deepvk/USER-bge-m3
export RERANKER_MODEL ?= qilowoq/bge-reranker-v2-m3-en-ru
export LLM_MODEL ?= qwen3.8:27b-mtp-q4_K_M

.DEFAULT_GOAL := help
.PHONY: help start build models pull-llm pull-hf up down restart logs ps gpu-check clean

help: ## Show the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-10s %s\n", $$1, $$2}'

start: build models up ## First run: build the image, download the models, start everything

build: ## Build the API image
	$(COMPOSE) build rag-api

models: pull-llm pull-hf ## Download all models; they are cached in volumes, so a rerun is fast

pull-llm: ## Pull the LLM into the ollama-models volume
	$(COMPOSE) up -d --wait ollama
	$(COMPOSE) exec ollama ollama pull $(LLM_MODEL)

# Loading the models (on CPU, without starting the API) downloads exactly the files they need.
pull-hf: ## Download the embedding and reranker models into the hf-cache volume
	$(COMPOSE) run --rm --no-deps rag-api python -c "from app.core.config import settings; \
		from sentence_transformers import CrossEncoder, SentenceTransformer; \
		SentenceTransformer(settings.RETRIEVER_MODEL, device='cpu'); \
		CrossEncoder(settings.RERANKER_MODEL, device='cpu')"

up: ## Start the stack and wait until every service is healthy
	$(COMPOSE) up -d --wait

down: ## Stop the stack; the database, documents and models are kept
	$(COMPOSE) down

restart: ## Rebuild and restart only the API after a code change
	$(COMPOSE) up -d --build --wait rag-api

logs: ## Follow the API and Ollama logs
	$(COMPOSE) logs -f rag-api ollama

ps: ## Show the services and their health
	$(COMPOSE) ps

gpu-check: ## Check that the API and Ollama run on the GPU
	$(COMPOSE) exec rag-api python -c "import torch; print('rag-api:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
	$(COMPOSE) exec ollama ollama ps

clean: ## Stop and DELETE all volumes: database, documents and downloaded models
	@read -p "This deletes the database, documents and models. Type yes: " answer && [ "$$answer" = yes ]
	$(COMPOSE) down -v
