# -*- coding: utf-8 -*-

from PyQt4 import QtCore


def clickable(widget):
    """Adds a clicked signal to a widget such as QLineEdit that
    ordinarily does not provide notifications of clicks."""

    class Filter(QtCore.QObject):

        clicked = QtCore.pyqtSignal()

        def eventFilter(self, obj, event):
            if obj == widget:
                if event.type() == QtCore.QEvent.MouseButtonRelease:
                    if obj.rect().contains(event.pos()):
                        self.clicked.emit()
                        return True
            return False

    filter = Filter(widget)
    widget.installEventFilter(filter)
    return filter.clicked
