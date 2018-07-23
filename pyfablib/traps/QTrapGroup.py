# -*- coding: utf-8 -*-

"""QTrapGroup.py: Container for optical traps."""

from pyqtgraph.Qt import QtCore, QtGui
from .QTrap import QTrap, states


class QTrapGroup(QtCore.QObject):

    def __init__(self, parent=None, name=None):
        super(QTrapGroup, self).__init__()
        self.ignoreUpdates = False
        self.parent = parent
        self.children = []
        self.name = name
        self._r = QtGui.QVector3D()

    def add(self, child):
        """Add a trap to the group"""
        child.parent = self
        self.children.append(child)

    def remove(self, thischild, delete=False):
        """Remove an object from the trap group.
        If the group is now empty, remove it
        from its parent group
        """
        if thischild in self.children:
            thischild.parent = None
            self.children.remove(thischild)
            if delete is True:
                thischild.deleteLater()
        else:
            for child in self.children:
                if isinstance(child, QTrapGroup):
                    child.remove(thischild, delete=delete)
        if ((len(self.children) == 0) and isinstance(self.parent, QTrapGroup)):
            self.parent.remove(self)

    def deleteLater(self):
        for child in self.children:
            child.deleteLater()
        super(QTrapGroup, self).deleteLater()

    def _update(self):
        if self.ignoreUpdates:
            return
        self.updatePosition()
        self.parent._update()

    def count(self):
        """Return the number of items in the group.
        """
        return len(self.children)

    def flatten(self):
        """Return a list of the traps in the group.
        """
        traps = []
        for child in self.children:
            if isinstance(child, QTrap):
                traps.append(child)
            else:
                traps.extend(child.flatten())
        return traps

    def isWithin(self, rect):
        """Return True if the entire group lies within
        the specified rectangle.
        """
        result = True
        for child in self.children:
            result = result and child.isWithin(rect)
        return result

    @property
    def state(self):
        """Current state of the children in the group.
        """
        return self.children[0].state

    @state.setter
    def state(self, state):
        for child in self.children:
            child.state = state

    def select(self, state=True):
        if state:
            self.state = states.selected
        else:
            self.state = states.normal

    @property
    def r(self):
        return self._r

    def updatePosition(self):
        self._r *= 0.
        traps = self.flatten()
        for trap in traps:
            self._r += trap.r
        self._r /= len(traps)

    def moveBy(self, dr):
        """Translate traps in the group.
        """
        self.ignoreUpdates = True
        # same displacement for all traps
        if isinstance(dr, QtGui.QVector3D):
            for child in self.children:
                child.moveBy(dr)
        # specified displacement for each trap
        else:
            for n, child in enumerate(self.children):
                child.moveBy(dr[n])
        self.ignoreUpdates = False
        self._update()

    def rotateTo(self, xy):
        """Rotate group of traps about its center.
        """
        pass
