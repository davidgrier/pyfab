# -*- coding: utf-8 -*-

"""QTrappingPattern.py: Interactive overlay for manipulating optical traps."""

from PyQt5.QtCore import (pyqtSignal, pyqtSlot, Qt,
                          QSize, QPoint, QRect, QRectF)
from PyQt5.QtGui import (QVector3D, QMouseEvent, QWheelEvent)
from PyQt5.QtWidgets import QRubberBand

import pyqtgraph as pg
from .QTrap import QTrap, states
from .QTrapGroup import QTrapGroup


class QTrappingPattern(pg.ScatterPlotItem):

    """Interface between QJansenScreen GUI and CGH pipeline.
    Implements logic for manipulating traps.
    """

    trapAdded = pyqtSignal(QTrap)
    sigCompute = pyqtSignal(object)

    def __init__(self, parent=None):
        super(QTrappingPattern, self).__init__()
        self.setParent(parent)  # this is not set by ScatterPlotItem
        self.setPxMode(False)   # scale plot symbols with window
        # Rubberband selection
        self.selection = QRubberBand(QRubberBand.Rectangle, self.parent())
        self.origin = QPoint()
        # traps, selected trap and active group
        self.pattern = QTrapGroup(self)
        self.trap = None
        self.group = None
        self.selected = []

    def refreshAppearance(self):
        """Provide a list of spots to screen for plotting.

        This will be called by children when their properties change.
        Changes can be triggered by mouse events, by interaction with
        property widgets, or by direct programmatic control of traps
        or groups.
        """
        traps = self.pattern.flatten()
        spots = [trap.spot for trap in traps]
        self.setData(spots=spots)
        return traps

    def refresh(self):
        traps = self.refreshAppearance()
        self.sigCompute.emit(traps)

    def selectedPoint(self, position):
        points = self.pointsAt(position)
        if not points:
            return None
        index = self.points().tolist().index(points[0])
        return index

    # Selecting traps and groups of traps
    def clickedTrap(self, pos):
        """Return the trap at the specified position
        """
        coords = self.mapFromScene(pos)
        index = self.selectedPoint(coords)
        if index is None:
            return None
        return self.pattern.flatten()[index]

    def groupOf(self, obj):
        """Return the highest-level group containing the specified object.
        """
        if obj is None:
            return None
        while obj.parent() is not self.pattern:
            obj = obj.parent()
        return obj

    def clickedGroup(self, pos):
        """Return the highest-level group containing the trap at
        the specified position.
        """
        self.trap = self.clickedTrap(pos)
        return self.groupOf(self.trap)

    def selectedTraps(self, region):
        """Return a list of traps whose groups fall
        entirely within the selection region.
        """
        self.selected = []
        rect = self.mapFromScene(QRectF(region)).boundingRect()
        for child in self.pattern.children():
            if child.isWithin(rect):
                self.selected.append(child)
                child.state = states.grouping
            else:
                child.state = states.normal
        self.refreshAppearance()

    # Creating and deleting traps
    def addTrap(self, trap):
        trap.setParent(self)
        trap.cgh = self.parent().cgh
        trap.state = states.selected
        self.pattern.add(trap)
        self.refresh()
        self.trapAdded.emit(trap)

    def createTrap(self, r):
        self.addTrap(QTrap(r=r))

    def createTraps(self, coordinates):
        coords = list(coordinates)
        if not coords:
            return
        self.pattern.blockRefresh = True
        group = QTrapGroup()
        self.pattern.add(group)
        for r in coords:
            trap = QTrap(r=r, parent=group)
            group.add(trap)
            self.trapAdded.emit(trap)
        self.pattern.blockRefresh = False
        self.refresh()
        return group

    def clearTraps(self):
        """Remove all traps from trapping pattern.
        """
        traps = self.pattern.flatten()
        for trap in traps:
            self.pattern.remove(trap, delete=True)
        self.refresh()

    # Creating, breaking and moving groups of traps
    def createGroup(self):
        """Combine selected objects into new group"""
        group = QTrapGroup()
        for trap in self.selected:
            if trap.parent() is not self:
                trap.parent().remove(trap)
            group.add(trap)
        self.pattern.add(group)
        self.selected = []

    def breakGroup(self):
        """Break group into children and
        place children in the top level.
        """
        if isinstance(self.group, QTrapGroup):
            for child in self.group.children():
                child.state = states.grouping
                self.group.remove(child)
                self.pattern.add(child)

    def moveGroup(self, pos):
        """Move the selected group so that the selected
        trap is at the specified position.
        """
        coords = self.mapFromScene(pos)
        dr = QVector3D(coords - self.trap.coords())
        self.group.moveBy(dr)

    # Dispatch low-level events to actions
    def leftPress(self, pos, modifiers):
        """Selection and grouping.
        """
        self.group = self.clickedGroup(pos)
        # update selection rectangle
        if self.group is None:
            self.origin = QPoint(pos)
            rect = QRect(self.origin, QSize())
            self.selection.setGeometry(rect)
            self.selection.show()
        # break selected group
        elif modifiers == Qt.ControlModifier:
            self.breakGroup()
        # select group
        else:
            self.group.state = states.selected
        self.refreshAppearance()

    def rightPress(self, pos, modifiers):
        """Creation and destruction.
        """
        # Shift-Right Click: Add trap
        if modifiers == Qt.ShiftModifier:
            self.createTrap(self.mapFromScene(pos))
        # Ctrl-Right Click: Delete trap
        elif modifiers == Qt.ControlModifier:
            self.pattern.remove(self.clickedGroup(pos), delete=True)
            self.refresh()

    # Handlers for signals emitted by QJansenScreen
    @pyqtSlot(QMouseEvent)
    def mousePress(self, event):
        """Event handler for mousePress events.
        """
        button = event.button()
        pos = event.pos()
        modifiers = event.modifiers()
        if button == Qt.LeftButton:
            self.leftPress(pos, modifiers)
        elif button == Qt.RightButton:
            self.rightPress(pos, modifiers)

    @pyqtSlot(QMouseEvent)
    def mouseMove(self, event):
        """Event handler for mouseMove events.
        """
        pos = event.pos()
        # Move traps
        if self.group is not None:
            self.moveGroup(pos)
        # Update selection box
        elif self.selection.isVisible():
            region = QRect(self.origin, QPoint(pos)).normalized()
            self.selection.setGeometry(region)
            self.selectedTraps(region)

    @pyqtSlot(QMouseEvent)
    def mouseRelease(self, event):
        """Event handler for mouseRelease events.
        """
        if self.selected:
            self.createGroup()
        for child in self.pattern.children():
            child.state = states.normal
        self.group = None
        self.selection.hide()
        self.refreshAppearance()

    @pyqtSlot(QWheelEvent)
    def mouseWheel(self, event):
        """Event handler for mouse wheel events.
        """
        pos = event.pos()
        group = self.clickedGroup(pos)
        if group is not None:
            group.state = states.selected
            dr = QVector3D(0., 0., event.delta() / 120.)
            group.moveBy(dr)
            # group.state = states.normal
        self.group = None
