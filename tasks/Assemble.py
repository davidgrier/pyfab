# -*- coding: utf-8 -*-

from .Task import Task

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Assemble(Task):
    """
    Takes the last QTrapGroup created by user and gives it
    to Pyfab's assembler. Subclassses will set the assembler's
    vertices.
    """

    def __init__(self, **kwargs):
        super(Assemble, self).__init__(**kwargs)

    def initialize(self, frame):
        self.cgh = self.parent.cgh.device
        self.assembler = self.parent.assembler
        # Set traps from last QTrapGroup created
        pattern = self.parent.pattern.pattern
        group = None
        for child in reversed(pattern.children()):
            if isinstance(child, type(pattern)):
                group = child
                break
        if group is None:
            logger.warning(
                "No traps selected. Please create a QTrapGroup.")
        self.assembler.traps = group
