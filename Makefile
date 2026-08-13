.PHONY: clean build leak-check twine-check upload

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

REPOSITORY ?= testpypi

clean:
	rm -rf dist

build: clean
	uv build --out-dir dist

leak-check: build
	@command -v unzip >/dev/null; \
	members="$$(unzip -Z1 dist/*.whl; tar -tzf dist/*.tar.gz)"; \
	test -n "$$members"; \
	if printf '%s\n' "$$members" | grep -E -q '(^|/)(glotscope-PRD\.md|HISTORY\.md|\.env($|\.)|[^/]+\.(pem|key|p12|pfx)|id_(rsa|dsa|ecdsa|ed25519)|\.netrc|\.pypirc|\.npmrc|\.ipynb_checkpoints/|[^/]+(~|\.sw[op])|\.git/|\.venv/|\.DS_Store|PROGRESS\.md|__pycache__/|[^/]+\.py[co])$$'; then \
		printf '%s\n' "Internal or sensitive path found in release artifacts" >&2; \
		exit 1; \
	fi

twine-check: leak-check
	uvx --from 'twine==7.0.0' twine check dist/*

upload: twine-check
	uvx --from 'twine==7.0.0' twine upload --repository "$(REPOSITORY)" dist/*
