import json
import os
import io
from PyQt4 import QtGui


class fabconfig(object):
    def __init__(self, parent):
        self.parent = parent
        fn = '~/.pyfab/pyfab.json'
        self.filename = os.path.expanduser(fn)

    def save(self, object):
        configuration = json.dumps(object.calibration,
                                   indent=2,
                                   separators=(',', ': '),
                                   ensure_ascii=False)
        with io.open(self.filename, 'w', encoding='utf8') as configfile:
            configfile.write(unicode(configuration))

    def restore(self, object):
        try:
            config = json.load(io.open(self.filename))
            object.calibration = config
        except IOError:
            print('could not open '+self.filename)

    def query_save(self, object):
        query = 'Save current configuration?'
        reply = QtGui.QMessageBox.question(self.parent,
                                           'Confirmation',
                                           query,
                                           QtGui.QMessageBox.Yes,
                                           QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            self.save(object)
        else:
            pass
