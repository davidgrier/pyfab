# -*- coding: utf-8 -*-

"""QTrapGroup.py: Container for optical traps."""

from PyQt5.QtCore import QObject
from PyQt5.QtGui import QVector3D
from .QTrap import QTrap, states


class QTrapGroup(QObject):

    def __init__(self, parent=None):
        super(QTrapGroup, self).__init__(parent)
        self._r = QVector3D()
        self.blockRefresh(False)

    # Organizing traps within the group
    def add(self, child):
        """Add a trap to the group"""
        child.setParent(self)

    def remove(self, thischild, delete=False):
        """Remove a trap from the group.
        If the group is now empty, remove it
        from its parent group
        """
        if thischild in self.children():
            thischild.setParent(None)
            if delete:
                thischild.deleteLater()
        else:
            for child in self.children():
                if isinstance(child, QTrapGroup):
                    child.remove(thischild, delete=delete)
        if self.empty() and isinstance(self.parent(), QTrapGroup):
            self.parent().remove(self, delete=True)

    def count(self):
        """Return the number of items in the group"""
        return len(self.children())

    def empty(self):
        """True if group has no children"""
        return self.count() == 0

    def flatten(self):
        """Return a list of the traps in the group"""
        return self.findChildren(QTrap)

    # Implementing changes in trap properties
    def blockRefresh(self, state):
        """Do not send refresh requests to parent if state is True"""
        self._blockRefresh = bool(state)

    def refreshBlocked(self):
        return self._blockRefresh

    def refresh(self):
        """Request parent to implement changes"""
        self.updatePosition()
        if not self.refreshBlocked():
            self.parent().refresh()

    # Methods for changing group properties
    def updatePosition(self):
        """The group is located at the center of mass of its children"""
        self._r *= 0.
        traps = self.flatten()
        for trap in traps:
            self._r += trap.r
        self._r /= len(traps)

    def moveBy(self, dr):
        """Translate traps in the group"""
        self.blockRefresh(True)
        # same displacement for all traps
        if isinstance(dr, QVector3D):
            for child in self.children():
                child.moveBy(dr)
        # specified displacement for each trap
        else:
            for n, child in enumerate(self.children()):
                child.moveBy(dr[n])
        self.blockRefresh(False)
        self.refresh()

    def moveTo(self, r):
        """Translate traps so that the group is centered at r"""
        dr = r - self.r
        self.moveBy(dr)

    def rotateTo(self, xy):
        """Rotate group of traps about its center"""
        pass

    def isWithin(self, rect):
        """Return True if the entire group lies within
        the specified rectangle.
        """
        result = True
        for child in self.children():
            result = result and child.isWithin(rect)
        return result

    def select(self, state=True):
        """Utility for setting state of group"""
        if state:
            self.state = states.selected
        else:
            self.state = states.normal

    # Group's properties
    @property
    def state(self):
        """Current state of the children in the group.
        """
        return self.children()[0].state

    @state.setter
    def state(self, state):
        for child in self.children():
            child.state = state

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, r):
        self.moveTo(r)
