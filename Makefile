# Prefer a real 3.12 if one is on PATH (Homebrew puts it at
# /opt/homebrew/bin/python3.12 on Apple silicon, /usr/local/bin on Intel), then
# fall back to whatever python3 is. Override with `make setup PY=/path/to/python`.
PY      ?= $(shell command -v python3.12 2>/dev/null || command -v python3 2>/dev/null || echo python3)
VENV    := .venv
BIN     := $(VENV)/bin
FIXTURE := fixtures/lofi-7.mp3
DEMO    := examples/lofi-7.house.mp3
REFS    ?= $(HOME)/Desktop/Mixpilot

.PHONY: help setup test demo learn clean lint

help:
	@echo "fourfloor"
	@echo "  make setup   create .venv and install fourfloor + dev deps"
	@echo "  make test    run the test suite (~40s)"
	@echo "  make demo    remix fixtures/lofi-7.mp3 into examples/"
	@echo "  make learn   derive a style profile from REFS=<folder of house remixes>"
	@echo "  make clean   remove build artefacts and generated audio"

setup:
	@$(PY) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || \
		{ echo "fourfloor needs Python 3.11+ ($(PY) is $$($(PY) -V 2>&1)). Try: make setup PY=/path/to/python3.12"; exit 1; }
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"
	@echo "ready: $(BIN)/fourfloor --help"

test:
	$(BIN)/python -m pytest

demo:
	@mkdir -p examples
	$(BIN)/python -m fourfloor remix $(FIXTURE) -o $(DEMO) --bpm 124 --length 4:30 --preview --no-wav
	@echo "wrote $(DEMO) + session, plan and preview.html"

learn:
	$(BIN)/python -m fourfloor learn "$(REFS)" -o styles/learned.json --anonymous

lint:
	$(BIN)/python -m compileall -q fourfloor tests

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -f examples/*.wav
