# Single entry point for both implementations. The Elixir and Python suites have
# to pass together: they share the corpus, so a change to one that the other does
# not agree with is a failure, not a detail.
PYTHON ?= python3
ELIXIR := elixir

.PHONY: test test-elixir test-python corpus format check clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

test: test-elixir test-python ## Run both suites

test-elixir: ## Elixir: format check, warnings-as-errors, tests
	cd $(ELIXIR) && mix format --check-formatted
	cd $(ELIXIR) && mix compile --force --warnings-as-errors
	cd $(ELIXIR) && mix test

test-python: ## Python: compile check, tests (incl. the cross-impl. corpus check)
	$(PYTHON) -m compileall -q gpark tests
	$(PYTHON) -m unittest discover -s tests -t . -v

corpus: ## Regenerate corpus specs and goldens from the Elixir kernels
	cd $(ELIXIR) && mix run -e '
	  for m <- [Gpark.Kernels.VecAddF32, Gpark.Kernels.SaxpyF32, Gpark.Kernels.ReduceSumF32] do
	    k = m.build()
	    {:ok, _} = Gpark.Validate.check(k)
	    File.write!("../corpus/specs/#{k.name}.json", Gpark.IR.JSON.encode!(k))
	    File.write!("../corpus/golden/#{k.name}.ptx", Gpark.PTX.emit(k))
	  end'

format: ## Format both implementations
	cd $(ELIXIR) && mix format
	$(PYTHON) -m ruff format gpark tests 2>/dev/null || true

check: test ## Alias for test

clean: ## Remove build artefacts
	cd $(ELIXIR) && mix clean
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
