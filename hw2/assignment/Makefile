EXTENSIONS := py sh png jpg txt pth pt PNG JPG
PY_SRC = $(wildcard src/*.py) $(wildcard src/**/*.py)
ALL_SRC = $(foreach ext,$(EXTENSIONS),$(wildcard src/*.$(ext)) $(wildcard src/**/*.$(ext)))
ALL_IMAGES = $(foreach ext,png jpg PNG JPG txt,$(wildcard images/*.$(ext)))
SUBMISSION_FILES = $(ALL_SRC) $(wildcard *.txt) pyproject.toml tox.ini $(ALL_IMAGES)
ASSIGNMENT_FILES = $(SUBMISSION_FILES) README.pdf Makefile requirements.txt

check-py-version:
	@if [ "$(shell python -c 'import sys; print(sys.version_info >= (3, 10))')" = "False" ]; then \
		echo "Please use Python 3.10 or newer"; \
		exit 1; \
	fi
.PHONY: check-py-version

setup: check-py-version
	python -m venv venv
	venv/Scripts/activate.bat && pip install --upgrade pip && pip install -r requirements.txt
.PHONY: setup



lint:
	flake8 $(PY_SRC)
.PHONY: lint

format:
	black $(PY_SRC)
.PHONY: format

submission.zip: $(SUBMISSION_FILES)
	"c:/Program Files/7-Zip/7z.exe" a submission.zip $(SUBMISSION_FILES)
	@echo "Zip done - upload submission.zip to Gradescope."

# --- commands below this line are for the course staff only ---

autograder.zip: $(wildcard autograder/*) $(wildcard tests/*) requirements.txt
	rm -f autograder/requirements.txt && ln -s "$(CURDIR)/requirements.txt" autograder/requirements.txt
	rm -rf autograder/tests && ln -s "$(CURDIR)/tests" autograder/tests
	rm -rf autograder/data && ln -s "$(CURDIR)/data" autograder/data
	rm -rf autograder/answerkey && ln -s "$(CURDIR)/answerkey" autograder/answerkey
	cd autograder && zip -x "*/__pycache__/*" "*/.DS_Store" ".DS_Store" -r ../autograder.zip * .[^.]*
	rm -f autograder/requirements.txt
	rm -rf autograder/tests
	rm -rf autograder/data
	rm -rf autograder/answerkey

answerkey.zip: $(wildcard answerkey/*.py)
	mkdir -p answerkey/src
	cp answerkey/*.py answerkey/src
	cp tox.ini pyproject.toml answerkey/
	cd answerkey && zip -r ../answerkey.zip src tox.ini pyproject.toml && rm -rf src && rm -f tox.ini pyproject.toml

README.pdf: README.md
	pandoc --template=eisvogel -o README.pdf README.md

assignment.zip: $(ASSIGNMENT_FILES)
	@echo "Creating assignment.zip containing files: $(ASSIGNMENT_FILES)"
	zip -r assignment.zip $(ASSIGNMENT_FILES)

clean:
	rm -f submission.zip autograder.zip README.pdf assignment.zip autograder/requirements.txt answerkey.zip results.json
	rm -rf autograder/tests answerkey/src
.PHONY: clean

zips: assignment.zip autograder.zip answerkey.zip
