.PHONY: help probe smoke report lint test survey build up clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

probe:  ## Report the local accelerator stack
	python scripts/gpu_probe.py

smoke:  ## Run the tiny training smoke test
	python scripts/smoke_train.py

report:  ## Generate a forum post from a real probe run (needs a GPU)
	python scripts/make_report.py

lint:  ## Lint the scripts and the scanner
	ruff check scripts/ rocm_portscan/ tests/ --extend-exclude tests/fixtures

test:  ## Run the scanner test suite
	python -m pytest -q

survey:  ## Rescan the 12 repositories in docs/SURVEY-2026-09.md
	scripts/run_survey.sh

build:  ## Build the ROCm image
	docker compose -f docker/docker-compose.yml build

up:  ## Run the probe inside the ROCm container
	docker compose -f docker/docker-compose.yml run --rm rocm

clean:  ## Remove local venvs and caches
	rm -rf .venv-test **/__pycache__ .ruff_cache
