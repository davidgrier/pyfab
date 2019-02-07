UIFILE := $(wildcard *.ui)
PYFILE := $(UIFILE:.ui=.py)
PYTHON = python
PYUIC = pyuic5

.PHONY: all test

all: $(PYFILE)

test: $(PYFILE)
	$(PYTHON) $(PYFILE)

%.py: %.ui
	$(PYUIC) $< -x -o $@
