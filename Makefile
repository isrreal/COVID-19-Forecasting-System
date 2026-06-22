.PHONY: test up down train

test:
	docker compose --profile test run --rm --build test

up:
	docker compose up -d

down:
	docker compose down

train:
	docker compose run --rm training python main_workflow.py --states $(STATES)
