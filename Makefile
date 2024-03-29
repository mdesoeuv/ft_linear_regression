VENV := .venv

PREDICT := predict.py
LEARN := learn.py


all: venv

venv: requirements.txt
			python3 -m venv $(VENV)
			. $(VENV)/bin/activate
			./$(VENV)/bin/pip install -r requirements.txt

clean:
	rm -rf $(VENV)
	rm -rf __pycache__

re: clean all

.PHONY: all, venv, clean, re