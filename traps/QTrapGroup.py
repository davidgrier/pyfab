#!/usr/bin/env python

"""QTrapGroup.py: Container for optical traps."""

from PyQt4 import QtCore
from QTrap import QTrap
from states import states


class QTrapGroup(QtCore.QObject):

    def __init__(self, parent=None, name=None, active=True):
        super(QTrapGroup, self).__init__()
        self.parent = parent
        self.children = []
        self.name = name
        self.active = active
        self.psi = None

    def add(self, child):
        """Add an object to the trap group.
        """
        child.parent = self
        child.active = self.active
        self.children.append(child)
        if child.psi is not None:
            if self.psi is None:
                self.psi = child.psi
            else:
                self.psi += child.psi

    def remove(self, thischild, delete=False):
        """Remove an object from the trap group.
        If the group is now empty, remove it
        from its parent group
        """
        self.psi = None
        if thischild in self.children:
            thischild.parent = None
            self.children.remove(thischild)
            if delete is True:
                thischild.deleteLater()
        else:
            for child in self.children:
                if isinstance(child, QTrapGroup):
                    child.remove(thischild, delete=delete)
        if (len(self.children) == 0) and (self.parent is not None):
            self.parent.remove(self)

    def deleteLater(self):
        for child in self.children:
            child.deleteLater()
        super(QTrapGroup, self).deleteLater()

    def update(self):
        if self.active:
            self.parent.update()

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
        if state in states:
            for child in self.children:
                child.state = state

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, active):
        for child in self.children:
            child.active = active
        self._active = active

    def moveBy(self, dr):
        """Translate traps in the group.
        """
        self.active = False
        for child in self.children:
            child.moveBy(dr)
        self.active = True
        self.update()
