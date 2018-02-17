from PyQt4 import QtGui, QtCore


def tabLayout(parent):
    layout = QtGui.QVBoxLayout(parent)
    layout.setAlignment(QtCore.Qt.AlignTop)
    layout.setSpacing(1)
    return layout
