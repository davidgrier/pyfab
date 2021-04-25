QRCFILE := help.qrc
RCFILE := $(QRCFILE:.qrc=_rc.py)
HELPFILES := $(wildcard html/*.html)

PYRCC = pyrcc5

UICOPTS = -x

.PHONY: all help clean

all: $(RCFILE)

help: help_rc.py

help_rc.py: help.qrc $(HELPFILES)

clean:
	-rm $(RCFILE)

%_rc.py: %.qrc
	$(PYRCC) $< -o $@
