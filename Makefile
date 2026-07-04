# ==============================================================================
# Developer entry points. Run `make help` for the catalogue.
# ==============================================================================
.DEFAULT_GOAL := help
SHELL := /bin/bash
PY ?= python

.PHONY: help install install-gpu install-tribe fmt lint typecheck test test-fast \
        migrate db-upgrade api frontend up down clean

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install the backend package + dev extras (CPU-only).
	$(PY) -m pip install -e ".[dev,explain,train]"

install-gpu:  ## Install the CUDA torch stack + GPU requirements.
	$(PY) -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	$(PY) -m pip install -r requirements-gpu.txt

install-tribe:  ## Clone + install the official Meta TRIBE v2 package.
	bash scripts/install_tribe.sh

fmt:  ## Auto-format and fix imports with ruff.
	ruff format backend tests
	ruff check --fix backend tests

lint:  ## Lint without modifying files.
	ruff check backend tests
	ruff format --check backend tests

typecheck:  ## Static type-check the backend.
	mypy backend/app

test:  ## Run the full test suite with coverage.
	pytest

test-fast:  ## Run tests excluding gpu/tribe-marked cases.
	pytest -m "not gpu and not tribe"

db-upgrade:  ## Apply Alembic migrations to head.
	cd backend && alembic upgrade head

migrate:  ## Autogenerate a new Alembic revision: make migrate m="message".
	cd backend && alembic revision --autogenerate -m "$(m)"

api:  ## Run the FastAPI dev server with reload.
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --app-dir backend

frontend:  ## Run the Vite dev server.
	cd frontend && npm run dev

up:  ## Start the full stack with Docker Compose (GPU).
	docker compose up --build

down:  ## Stop the Docker Compose stack.
	docker compose down

clean:  ## Remove caches and build artifacts.
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
