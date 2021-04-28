QRCFILE := $(wildcard *.qrc)
RCFILE := $(QRCFILE:.qrc=_rc.py)
HELPFILES := $(wildcard html/*.html)

PYRCC = pyrcc5

.PHONY: all help clean

all: help

help: $(RCFILE)

clean:
	-rm $(RCFILE)

%_rc.py: %.qrc
	$(PYRCC) $< -o $@
