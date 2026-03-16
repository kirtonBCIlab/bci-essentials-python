## Makefile for BCI Essentials Python package. Requires make installed on the system. This is available by default on Linux and macOS, and can be installed on Windows via WSL or other means.
.PHONY: help install dev-install test lint format

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

install: ## Install package and dev dependencies
	pip install .

dev-install: ## Install development dependencies (black, flake8)
	pip install -e .
	pip install black flake8

test: ## Run tests
	python -m unittest

lint: ## Run black and flake8
	black --check .
	flake8

format: ## Auto-format code with black
	black .
