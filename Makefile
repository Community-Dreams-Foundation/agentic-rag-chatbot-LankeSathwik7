.PHONY: sanity

sanity:
	@mkdir -p artifacts
	@{ command -v python3 >/dev/null 2>&1 && PY=python3 || PY=python; $$PY -m agentic_rag.sanity > artifacts/sanity_run.log; }
	@echo "Generated artifacts/sanity_output.json"
