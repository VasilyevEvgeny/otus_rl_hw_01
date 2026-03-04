IMAGE_NAME := otus_rl_hw_01
CONTAINER_NAME := otus_rl_hw_01

.PHONY: build up down attach shell

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

attach:
	docker exec -it $(CONTAINER_NAME) bash

shell: build up attach
