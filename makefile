# Install dependencies
install:
	python3 -m venv venv
	./venv/bin/pip install -r requirements.txt

# Run Flask server
run:
	FLASK_APP=app.py FLASK_ENV=development ./venv/bin/flask run --port 3000
