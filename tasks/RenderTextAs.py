# -*- coding: utf-8 -*-
# MENU: Add trap/Render text ...

from .RenderText import RenderText
from PyQt5.QtGui import QInputDialog


class RenderTextAs(RenderText):
    """Render user-selected text as a pattern of traps"""

    def __init__(self, **kwargs):
        super(RenderTextAs, self).__init__(**kwargs)
        qtext, ok = QInputDialog.getText(self.parent,
                                         'Render Text',
                                         'Text:')
        if ok:
            self.text = str(qtext)
        else:
            self.text = 'hello'
