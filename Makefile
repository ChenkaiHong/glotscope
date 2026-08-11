.PHONY: clean build leak-check twine-check upload

REPOSITORY ?= testpypi

clean:
	rm -rf dist

build: clean
	uv build --out-dir dist

leak-check: build
	@members="$$(unzip -Z1 dist/*.whl; tar -tzf dist/*.tar.gz)"; \
	if printf '%s\n' "$$members" | rg -q '(^|/)(glotscope-PRD\.md|HISTORY\.md|\.env|[^/]+\.(pem|key)|__pycache__/|[^/]+\.py[co])$$'; then \
		printf '%s\n' "Internal or sensitive path found in release artifacts" >&2; \
		exit 1; \
	fi

twine-check: leak-check
	uvx --from 'twine==7.0.0' twine check dist/*

upload: twine-check
	uvx --from 'twine==7.0.0' twine upload --repository "$(REPOSITORY)" dist/*
