import json
import os
import io
from datetime import datetime
from PyQt4 import QtGui


class fabconfig(object):
    def __init__(self, parent):
        self.parent = parent
        self.datadir = os.path.expanduser('~/data/')
        self.configdir = os.path.expanduser('~/.pyfab/')
        self.configfile = os.path.join(self.configdir, 'pyfab.json')

    def timestamp(self):
        return datetime.now().strftime('_%Y%b%d_%H%M%S')
    
    def filename(self, prefix='pyfab', suffix=None):
        return os.path.join(self.datadir,
                            prefix + self.timestamp() + suffix)

    def save(self, object):
        configuration = json.dumps(object.calibration,
                                   indent=2,
                                   separators=(',', ': '),
                                   ensure_ascii=False)
        with io.open(self.configfile, 'w', encoding='utf8') as configfile:
            configfile.write(unicode(configuration))

    def restore(self, object):
        try:
            config = json.load(io.open(self.configfile))
            object.calibration = config
        except IOError:
            print('could not open '+self.configfile)

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
