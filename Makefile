# Single entry point for both implementations. The Elixir and Python suites have
# to pass together: they share the corpus, so a change to one that the other does
# not agree with is a failure, not a detail.
PYTHON ?= python3
ELIXIR := elixir
PYTHON_DIR := python

.PHONY: test test-elixir test-python corpus remote remote-build format check clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

test: test-elixir test-python ## Run both suites

test-elixir: ## Elixir: format check, warnings-as-errors, tests
	cd $(ELIXIR) && mix format --check-formatted
	cd $(ELIXIR) && mix compile --force --warnings-as-errors
	cd $(ELIXIR) && mix test

test-python: ## Python: compile check, tests (incl. the cross-impl. corpus check)
	cd $(PYTHON_DIR) && $(PYTHON) -m compileall -q gpark tests
	cd $(PYTHON_DIR) && $(PYTHON) -m unittest discover -s tests -t . -v

corpus: ## Regenerate corpus specs and goldens from the Elixir kernels
	cd $(ELIXIR) && mix run regen_corpus.exs

remote: ## Assemble every golden for ARCH (needs ptxas) and report spills
	GPARK_ARCH=$${GPARK_ARCH:-sm_80} remote/ptxas_check.sh "$$GPARK_ARCH"

remote-build: ## Build the CUDA harnesses (needs nvcc + -lcuda)
	nvcc -O2 -o remote/exec_harness remote/exec_harness.cu -lcuda
	nvcc -O2 -o remote/graph_bench remote/graph_bench.cu -lcuda

format: ## Format both implementations
	cd $(ELIXIR) && mix format
	cd $(PYTHON_DIR) && $(PYTHON) -m ruff format gpark tests 2>/dev/null || true

check: test ## Alias for test

clean: ## Remove build artefacts
	cd $(ELIXIR) && mix clean
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
