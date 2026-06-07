PY ?= python3
VENV ?= .venv
VENV_PY = $(VENV)/bin/python

.PHONY: venv install fixtures ingest smoke smoke-offline clean

# Create an isolated virtualenv and install deps into it.
venv:
	$(PY) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt
	@echo "venv ready — activate with: source $(VENV)/bin/activate"

install:
	$(PY) -m pip install -r requirements.txt

fixtures:
	$(PY) -m catalogue_intel.fixtures.generate_fixtures

ingest:
	$(PY) -m catalogue_intel.ingest

# Default smoke hits the REAL OpenAI API (the actual end-to-end test, PRD §7).
smoke:
	$(PY) -m catalogue_intel.smoke.run_smoke

# Fast structural checks with a stubbed client — no key, no network.
smoke-offline:
	SMOKE_OFFLINE=1 $(PY) -m catalogue_intel.smoke.run_smoke

clean:
	rm -rf catalogue_intel/data
