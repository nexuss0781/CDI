# CDI Testing Guide

## Commands to Run

### 1. Syntax Check (No Dependencies Required)
```bash
python tests/check_syntax.py
```

### 2. Integration Tests (Requires Dependencies)
```bash
python tests/integration_tokenizer.py
```

### 3. Run All Tests
```bash
python run_tests.py
```

## Before Running Tests

Install dependencies:
```bash
pip install -r requirements.txt
```

## Test Structure

- `tests/check_syntax.py` — Validates Python syntax for all files (no deps)
- `tests/integration_tokenizer.py` — Full integration tests (requires torch, ethiobbpe)
- `run_tests.py` — Main orchestrator at root level

## That's it.

Everything is organized. Tests are in `tests/`. Orchestrator is at root.
