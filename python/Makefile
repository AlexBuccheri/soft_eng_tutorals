# Makefile, based on https://github.com/AlexBuccheri/python_cookie_cutter

# Specify path to virtual environment to use for this project
VENV_PATH := ../venv

# Ensures all commands for a given target are run in the same shell
# Facilitates running installations in the specified venv
.ONESHELL:

# Default command for `make`
.DEFAULT_GOAL: install-dev

# Targets that do not correspond to files
.PHONY: install install-dev format check-format cov

# Production installation
install: pyproject.toml
	. $(VENV_PATH)/bin/activate
	python -m pip install --upgrade pip setuptools wheel
	python -m pip install .

# Development installation
# Install dependencies specified in .dev of pyproject.toml
install-dev: pyproject.toml
	. $(VENV_PATH)/bin/activate
	python -m pip install --upgrade pip setuptools wheel
	python -m pip install -e ".[dev]"

# Apply formatting
format:
	isort src/ test/
	black src/${PACKAGE}
	black test/
	ruff check src/${PACKAGE}
	ruff check test/

# Check formatting
check-format:
	ruff check src/ --output-format=github
	ruff check test/ --output-format=github
	isort --check src/ test/
	black --check src/
	black --check test/

# Run codecov
cov:
	pytest --cov=src/ --cov-report term --cov-fail-under=-1
