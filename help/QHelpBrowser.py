from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
import os


class QHelpBrowser(QWebEngineView):

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
