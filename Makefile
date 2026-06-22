.PHONY: test up down train

test:
	docker compose run --rm api sh -c "pip install -r requirements/dev.txt -q && python -m pytest tests/ -v --tb=short"

up:
	docker compose up -d

down:
	docker compose down

train:
	docker compose run --rm training python main_workflow.py --states $(STATES)
