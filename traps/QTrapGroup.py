#!/usr/bin/env python

"""QTrapGroup.py: Container for optical traps."""

from QTrap import QTrap
from states import states


class QTrapGroup(object):

    def __init__(self, name=None):
        self.parent = None
        self.children = []
        self.name = name

    def add(self, child):
        """Add an object to the trap group.
        """
        self.children.append(child)
        child.parent = self

    def remove(self, thischild):
        """Remove an object from the trap group.
        If the group is now empty, remove it
        from its parent group
        """
        if thischild in self.children:
            thischild.parent = None
            self.children.remove(thischild)
        else:
            for child in self.children:
                if isinstance(child, QTrapGroup):
                    child.remove(thischild)
        if (len(self.children) == 0) and (self.parent is not None):
            self.parent.remove(self)

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

    def moveBy(self, dr):
        """Translate traps in the group.
        """
        for child in self.children:
            child.moveBy(dr)
