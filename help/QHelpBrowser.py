# -*- coding: utf-8 -*-

try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView as Browser
except ImportError:
    from PyQt5.QtWebKitWidgets import QWebView as Browser
from PyQt5.QtCore import QUrl
import os


class QHelpBrowser(Browser):
    '''HTML Widget for reading documentation

    Methods
    -------
    load(document)
        Load the specified document

        Arguments
        ---------
        document: str
            name of the html document stored in the help directory
    '''

    def __init__(self, basename):
        super(QHelpBrowser, self).__init__()

        self.dir = os.path.dirname(__file__)
        self.load(basename)
        self.show()

    def load(self, basename):
        filename = os.path.join(self.dir, basename + '.html')
        path = os.path.abspath(filename)
        url = QUrl.fromLocalFile(path)
        super(QHelpBrowser, self).load(url)
