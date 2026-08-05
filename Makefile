.PHONY: up down logs build test lint dev strikerdb seed

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

build:
	docker compose build

test:
	uv run pytest

lint:
	uv run python -m compileall -q app

dev:
	uv run uvicorn app.main:app --reload --port 8000

# Build the StrikeDB binary locally (bare-metal dev / verification)
strikerdb:
	cd reference/StrikeDB && cargo build --release --bin dbstrike

# Seed the index with a few starter queries (needs the API running)
seed:
	curl -s -X POST http://localhost:8000/v1/admin/ingest \
	  -H "Authorization: Bearer $(BOOTSTRAP_ADMIN_KEY)" \
	  -H "Content-Type: application/json" \
	  -d '{"queries": ["aurora borealis wallpaper", "macro photography butterfly", "vintage camera collection", "neon city night street", "minimalist architecture facade", "wildlife lion portrait"]}' \
	  && echo " seeded"
