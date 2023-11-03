version = $(shell cat package.json | grep version | awk -F'"' '{print $$4}')

install:
	pip3 install poetry
	poetry install

run:
	poetry run python3 main.py

