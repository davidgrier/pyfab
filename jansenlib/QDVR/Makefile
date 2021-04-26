QRCFILE := $(wildcard *.qrc)
RCFILE := $(QRCFILE:.qrc=_rc.py)

PYRCC = pyrcc5

.PHONY: all clean

all: $(RCFILE)

clean:
	-rm $(RCFILE)

%_rc.py: %.qrc
	$(PYRCC) $< -o $@
