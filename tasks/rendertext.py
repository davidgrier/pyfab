from PIL import Image, ImageDraw, ImageFont
import numpy as np
from task import task
from PyQt4.QtGui import QVector3D
import os


class rendertext(task):

    def __init__(self, txt='hello', **kwargs):
        super(rendertext, self).__init__(**kwargs)
        dir, _ = os.path.split(__file__)
        font = os.path.join(dir, 'Ubuntu-R.ttf')
        self.font = ImageFont.truetype(font)
        self.text = txt

    def dotask(self):
        sz = self.font.getsize(self.text)
        img = Image.new('L', sz, 0)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), self.text, font=self.font, fill=255)
        bmp = np.flip(np.array(img), 0) > 128
        y, x = np.nonzero(bmp)
        x = (x - np.mean(x)) * 15 + 320
        y = (y - np.mean(y)) * 15 + 240
        p = list(map(lambda x, y: QVector3D(x, y, 0), x, y))
        print('CREATING TRAPS ...')
        self.parent.pattern.createTraps(p)
