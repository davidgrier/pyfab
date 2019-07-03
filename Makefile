UIFILE := $(wildcard *.ui)
PYFILE := $(UIFILE:.ui=.py)
QRCFILE := $(wildcard *.qrc)
RCFILE := $(QRCFILE:.qrc=_rc.py)
HELPFILES := $(wildcard help/*.html)

PYTHON = python
PYUIC = pyuic5
PYRCC = pyrcc5

UICOPTS = -x

.PHONY: all help test clean

all: $(PYFILE) $(RCFILE) help

help: help.qrc $(HELPFILES)
	$(PYRCC) help.qrc -o help_rc.py

test: $(PYFILE)
	$(PYTHON) $(PYFILE)

clean:
	-rm $(PYFILE) $(RCFILE)

%.py: %.ui
	$(PYUIC) $< $(UICOPTS) -o $@

%_rc.py: %.qrc
	$(PYRCC) $< -o $@
