.PHONY: download shards train label export test

download:
	bash scripts/download_librispeech.sh

shards:
	python -m biona_lab.data.shards

train:
	python -m biona_lab.training.train

label:
	python scripts/run_labeling.py

export:
	python scripts/export_model.py

test:
	pytest tests/
