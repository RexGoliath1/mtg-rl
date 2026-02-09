.PHONY: ci lint test docker-build clean help sync-data test-train

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

S3_BUCKET := mtg-rl-checkpoints-20260124190118616600000001

sync-data: ## Download latest collection data from S3
	@echo "Finding latest collection..."
	@LATEST=$$(aws s3 ls s3://$(S3_BUCKET)/imitation_data/ | grep PRE | awk '{print $$2}' | sort | tail -1) && \
	if [ -z "$$LATEST" ]; then echo "No collections found in S3"; exit 1; fi && \
	echo "Syncing: $$LATEST" && \
	mkdir -p data/test_training && \
	aws s3 sync "s3://$(S3_BUCKET)/imitation_data/$$LATEST" data/test_training/ --exclude "*.log" --exclude "*.txt" && \
	echo "\n✓ Data synced to data/test_training/" && \
	ls -lh data/test_training/*.h5 2>/dev/null || echo "  (no HDF5 files yet — collection may still be running)"

test-train: ## Quick 1-epoch local training test on synced data
	WANDB_MODE=disabled uv run --extra training python3 scripts/train_imitation.py \
		--data-dir data/test_training --epochs 1 --output /tmp/test_train.pt
	@echo "\n✓ Local training test passed"

clean: ## Remove build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
