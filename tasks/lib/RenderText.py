# -*- coding: utf-8 -*-
# MENU: Add trap/Render text

from ..QTask import QTask
import numpy as np
from numpy.random import normal
from PyQt5.QtGui import (QPixmap, QPainter, QPen, QColor, QFont, QVector3D)


class RenderText(QTask):
    """Render text as a pattern of traps"""

    def __init__(self,
                 text='hello',
                 spacing=20,
                 fuzz=0.05,
                 **kwargs):
        super(RenderText, self).__init__(**kwargs)
        self.spacing = spacing
        self.fuzz = fuzz
        self.text = text

    def get_coordinates(self):
        '''Returns coordinates of lit pixels in text'''
        if len(self.text) == 0:
            return []
        # Buffer for drawing text
        w, h = 100, 20
        pixmap = QPixmap(w, h)
        pixmap.fill(QColor('black'))

        # Painter for rendering text
        painter = QPainter(pixmap)

        # White text on black background
        pen = QPen()
        pen.setWidth(1)
        pen.setColor(QColor('white'))
        painter.setPen(pen)

        # Small sans serif font, no antialiasing
        font = QFont()
        font.setFamily('Arial')
        font.setPointSize(9)
        font.setStyleStrategy(QFont.NoAntialias)
        painter.setFont(font)

        # Draw text into buffer
        painter.drawText(0, h, self.text)
        painter.end()

        # Extract bitmap data from buffer
        image = pixmap.toImage()
        data = image.bits()
        data.setsize(h*w*4)
        bmp = np.frombuffer(data, np.uint8).reshape((h, w, 4))
        bmp = bmp[:,:,0]

        # Coordinates of lit pixels
        y, x = np.nonzero(bmp)
        x = x + normal(scale=self.fuzz, size=len(x)) - np.mean(x)
        y = y + normal(scale=self.fuzz, size=len(y)) - np.mean(y)
        x *= self.spacing
        y *= self.spacing
        x += self.parent().camera.device.width/2
        y += self.parent().camera.device.height/2
        p = list(map(lambda x, y: QVector3D(x, y, 0), x, y))
        return p

    def complete(self):
        p = self.get_coordinates()
        group = self.parent().pattern.createTraps(p)
        self.setData({'traps': group})
