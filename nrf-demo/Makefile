install:
	pip install -r requirements.txt
	pip install .

run:
	chainlit run main.py -h --host "0.0.0.0" --port 5050


docker:
	docker build -t nrf-chainlit-ui .

up:
	docker compose up -d --build

down:
	docker compose down