.PHONY: ci lint test docker-build clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

ci: lint test ## Run full CI pipeline (lint + test)
	@echo "\n✓ CI pipeline passed"

lint: ## Run ruff linter
	uv run python3 -m ruff check . --ignore E501 --exclude forge-repo,wandb

test: ## Run parser tests
	uv run python3 -m pytest tests/test_parser.py -v

docker-build: ## Build Docker images (no push)
	docker build -f infrastructure/docker/Dockerfile.collection -t mtg-collection:local .
	docker build -f infrastructure/docker/Dockerfile.training --build-arg BASE_IMAGE=python:3.11-slim -t mtg-training:local .
	@echo "\n✓ Docker builds passed"

docker-integration: docker-build ## Run full Docker integration test locally
	@echo "Starting daemon..."
	docker build -f infrastructure/docker/Dockerfile.daemon -t mtg-daemon:local .
	docker rm -f mtg-daemon-local mtg-collection-local 2>/dev/null || true
	docker run -d --name mtg-daemon-local -p 17171:17171 mtg-daemon:local
	@echo "Waiting for daemon to start (60s)..."
	@sleep 60
	@echo "Running collector (5 games)..."
	docker run --name mtg-collection-local \
		--link mtg-daemon-local:daemon \
		-e PYTHONUNBUFFERED=1 \
		mtg-collection:local \
		python -u scripts/collect_ai_training_data.py \
			--games 5 --workers 1 --host daemon --port 17171 --timeout 60
	@echo "Cleaning up..."
	docker rm -f mtg-daemon-local mtg-collection-local 2>/dev/null || true
	@echo "\n✓ Docker integration test passed"

clean: ## Remove build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
