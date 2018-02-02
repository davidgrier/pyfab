#!/usr/bin/env python

"""QTrappingPattern.py: Interface between QJansenScreen and QSLM."""

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from QTrap import QTrap, state
from QTrapGroup import QTrapGroup


class QTrappingPattern(pg.ScatterPlotItem):
    """Interface between QJansenScreen GUI and CGH pipeline.
    Implements logic for manipulating traps.
    """

    trapAdded = QtCore.pyqtSignal(QTrap)

    def __init__(self, parent=None, pipeline=None):
        super(QTrappingPattern, self).__init__()
        self.parent = parent
        self.pipeline = pipeline
        self.pattern = QTrapGroup(parent=self)
        self.parent.addOverlay(self)

        # Connect to signals coming from screen
        self.parent.sigMousePress.connect(self.mousePress)
        self.parent.sigMouseMove.connect(self.mouseMove)
        self.parent.sigMouseRelease.connect(self.mouseRelease)
        self.parent.sigMouseWheel.connect(self.mouseWheel)
        self.pipeline.sigComputing.connect(self.pauseSignals)
        # Rubberband selection
        self.selection = QtGui.QRubberBand(
            QtGui.QRubberBand.Rectangle, self.parent)
        self.origin = QtCore.QPoint()
        # traps, selected trap and active group
        self.trap = None
        self.group = None
        self.selected = []

    def pauseSignals(self, pause):
        self.parent.emitSignals = not pause

    def _update(self, project=True):
        """Provide a list of spots to screen for plotting
        and optionally send trap data to CGH pipeline.

        This will be called by children when their properties change.
        Changes can be triggered by mouse events, by interaction with
        property widgets, or by direct programmatic control of traps
        or groups.
        """
        traps = self.pattern.flatten()
        spots = [trap.spot for trap in traps]
        self.setData(spots=spots)
        if project and self.pipeline is not None:
            self.pipeline.traps = traps
            self.pipeline.compute()

    def dataCoords(self, pos):
        """Convert pixel position in screen widget to
        image coordinates.
        """
        return self.mapFromScene(pos)

    def selectedPoint(self, position):
        points = self.pointsAt(position)
        if len(points) <= 0:
            return None
        index = self.points().tolist().index(points[0])
        return index

    # Selecting traps and groups of traps
    def clickedTrap(self, pos):
        """Return the trap at the specified position
        """
        coords = self.dataCoords(pos)
        index = self.selectedPoint(coords)
        if index is None:
            return None
        return self.pattern.flatten()[index]

    def groupOf(self, obj):
        """Return the highest-level group containing the specified object.
        """
        if obj is None:
            return None
        while obj.parent is not self.pattern:
            obj = obj.parent
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
        rect = self.dataCoords(QtCore.QRectF(region)).boundingRect()
        for child in self.pattern.children:
            if child.isWithin(rect):
                self.selected.append(child)
                child.state = state.grouping
            else:
                child.state = state.normal
        if len(self.selected) <= 1:
            self.selected = []
        self._update(project=False)

    # Creating and deleting traps
    def createTrap(self, pos, update=True):
        trap = QTrap(r=self.dataCoords(pos), parent=self)
        self.pattern.add(trap)
        self.trapAdded.emit(trap)
        if update:
            self._update()

    def createTraps(self, coordinates):
        coords = list(coordinates)
        if len(coords) < 1:
            return
        group = QTrapGroup(active=False)
        self.pattern.add(group)
        for r in coords:
            trap = QTrap(r=r, parent=group, active=False)
            group.add(trap)
            self.trapAdded.emit(trap)
        group.active = True
        self._update()

    def clearTraps(self):
        """Remove all traps from trapping pattern.
        """
        traps = self.pattern.flatten()
        for trap in traps:
            self.pattern.remove(trap, delete=True)
        self._update()

    # Creating, breaking and moving groups of traps
    def createGroup(self):
        """Combine selected objects into new group.
        """
        if len(self.selected) == 0:
            return
        group = QTrapGroup()
        for trap in self.selected:
            if trap.parent is not self:
                trap.parent.remove(trap)
            group.add(trap)
        self.pattern.add(group)
        self.selected = []

    def breakGroup(self):
        """Break group into children and
        place children in the top level.
        """
        if isinstance(self.group, QTrapGroup):
            for child in self.group.children:
                child.state = state.grouping
                self.group.remove(child)
                self.pattern.add(child)

    def moveGroup(self, pos):
        """Move the selected group so that the selected
        trap is at the specified position.
        """
        coords = self.dataCoords(pos)
        dr = QtGui.QVector3D(coords - self.trap.coords())
        self.group.moveBy(dr)

    # Dispatch low-level events to actions
    def leftPress(self, pos, modifiers):
        """Selection and grouping.
        """
        self.group = self.clickedGroup(pos)
        # update selection rectangle
        if self.group is None:
            self.origin = QtCore.QPoint(pos)
            rect = QtCore.QRect(self.origin, QtCore.QSize())
            self.selection.setGeometry(rect)
            self.selection.show()
        # break selected group
        elif modifiers == QtCore.Qt.ControlModifier:
            self.breakGroup()
        # select group
        else:
            self.group.state = state.selected
        self._update(project=False)

    def rightPress(self, pos, modifiers):
        """Creation and destruction.
        """
        # Shift-Right Click: Add trap
        if modifiers == QtCore.Qt.ShiftModifier:
            self.createTrap(pos)
        # Ctrl-Right Click: Delete trap
        elif modifiers == QtCore.Qt.ControlModifier:
            self.pattern.remove(self.clickedGroup(pos), delete=True)
            self._update()

    # Handlers for signals emitted by QJansenScreen
    @QtCore.pyqtSlot(QtGui.QMouseEvent)
    def mousePress(self, event):
        """Event handler for mousePress events.
        """
        button = event.button()
        pos = event.pos()
        modifiers = event.modifiers()
        if button == QtCore.Qt.LeftButton:
            self.leftPress(pos, modifiers)
        elif button == QtCore.Qt.RightButton:
            self.rightPress(pos, modifiers)

    @QtCore.pyqtSlot(QtGui.QMouseEvent)
    def mouseMove(self, event):
        """Event handler for mouseMove events.
        """
        pos = event.pos()
        # Move traps
        if self.group is not None:
            self.moveGroup(pos)
        # Update selection box
        elif self.selection.isVisible():
            region = QtCore.QRect(self.origin, QtCore.QPoint(pos)).normalized()
            self.selection.setGeometry(region)
            self.selectedTraps(region)

    @QtCore.pyqtSlot(QtGui.QMouseEvent)
    def mouseRelease(self, event):
        """Event handler for mouseRelease events.
        """
        self.createGroup()
        for child in self.pattern.children:
            child.state = state.normal
        self.group = None
        self.selection.hide()
        self._update(project=False)

    @QtCore.pyqtSlot(QtGui.QWheelEvent)
    def mouseWheel(self, event):
        """Event handler for mouse wheel events.
        """
        pos = event.pos()
        group = self.clickedGroup(pos)
        if group is not None:
            group.state = state.selected
            dr = QtGui.QVector3D(0., 0., event.delta() / 120.)
            group.moveBy(dr)
            group.state = state.normal
        self.group = None
