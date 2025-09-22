# Basic Reference RAG Implementation - Simple Makefile

.PHONY: help install lint format test clean run setup

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation and setup
install: ## Install all dependencies
	uv sync

setup: install ## Complete development setup
	uv run pre-commit install
	@echo "✅ Development environment ready!"
	@echo "Run 'make run' to start the application"

# Code quality
lint: ## Run linting with auto-fix
	uv run ruff check --fix .

format: ## Format code
	uv run ruff format .

check: lint format ## Run all code quality checks

# Testing
test: ## Run tests
	uv run pytest tests/ -v

# Application
run: ## Run the Chainlit web interface
	uv run chainlit run src/chainlit_app.py

dev: ## Run LangGraph Studio for development
	uv run langgraph dev

# Database setup
setup-structured-db: ## Initialize databases with sample data
	uv run python scripts/structured_data_ingestion_pipeline.py

# Embeddings setup
setup-embeddings-db: ## Initialize databases with sample data
	uv run python scripts/unstructured_data_ingestion_pipeline.py


# All setup
setup-db: setup-structured-db setup-embeddings-db

# Cleanup
clean: ## Clean up cache and temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

# Pre-commit
precommit: ## Run pre-commit on all files
	uv run pre-commit run --all-files

# Quick development workflow
quick: clean lint format test ## Quick development check (clean, lint, format, test)

# First-time setup
first-setup: ## Complete first-time setup
	uv sync
	uv run pre-commit install
	python scripts/structured_data_ingestion_pipeline.py || echo "⚠️  Database setup skipped - run 'make setup-db' later"
	python scripts/unstructured_data_ingestion_pipeline.py || echo "⚠️  Database setup skipped - run 'make setup-db' later"
	@echo ""
	@echo "🎉 Setup complete! Next steps:"
	@echo "   make run    # Start the web interface"
	@echo "   make dev    # Start LangGraph Studio"
	@echo "   make test   # Run tests"
