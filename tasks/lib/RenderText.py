# -*- coding: utf-8 -*-
# MENU: Add trap/Render text

from ..QTask import QTask
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from numpy.random import normal
from PyQt5.QtGui import QVector3D
import os


class RenderText(QTask):
    """Render text as a pattern of traps"""

    def __init__(self,
                 text='hello',
                 spacing=20,
                 fuzz=0.05,
                 **kwargs):
        super(RenderText, self).__init__(**kwargs)
        dir, _ = os.path.split(__file__)
        font = os.path.join(dir, 'Ubuntu-R.ttf')
        self.font = ImageFont.truetype(font)
        self.spacing = spacing
        self.fuzz = fuzz
        self.text = text

    def get_coordinates(self):
        if len(self.text) == 0:
            return []
        sz = self.font.getsize(self.text)
        img = Image.new('L', sz, 0)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), self.text, font=self.font, fill=255)
        bmp = np.array(img) > 128
        bmp = bmp[::-1]
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
        self.parent().pattern.createTraps(p)
