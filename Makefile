.PHONY: setup test kaggle-push-src kaggle-run kaggle-pull

## setup: Create virtual environment and install requirements
setup:
	uv venv .venv
	.venv/bin/pip install -r requirements.txt

## test: Run pytest
test:
	.venv/bin/python -m pytest -q

## kaggle-push-src: Push source dataset to Kaggle
kaggle-push-src:
	python scripts/push_source_dataset.py

## kaggle-run: Run training on Kaggle
kaggle-run:
	python scripts/run_on_kaggle.py

## kaggle-pull: Pull results from Kaggle
kaggle-pull:
	python scripts/run_on_kaggle.py --pull-only
