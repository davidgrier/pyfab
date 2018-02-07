from PyQt4 import QtCore, QtWebKit
import os


class QHelpBrowser(QtWebKit.QWebView):

    def __init__(self, basename):
        super(QHelpBrowser, self).__init__()

        self.dir = os.path.dirname(__file__)
        self.load(basename)
        self.show()

    def load(self, basename):
        filename = os.path.join(self.dir, basename + '.html')
        path = os.path.abspath(filename)
        url = QtCore.QUrl.fromLocalFile(path)
        super(QHelpBrowser, self).load(url)
