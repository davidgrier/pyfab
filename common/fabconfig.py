# -*- coding: utf-8 -*-

import json
import os
import io
from datetime import datetime
from PyQt4 import QtGui
import logging


class fabconfig(object):
    """Save and restore configuration of objects for pyfab/jansen."""

    def __init__(self, parent):
        self.parent = parent
        self.datadir = os.path.expanduser('~/data/')
        self.configdir = os.path.expanduser('~/.pyfab/')
        if not os.path.exists(self.datadir):
            logging.info('Creating data directory: ' + self.datadir)
            os.makedirs(self.datadir)
        if not os.path.exists(self.configdir):
            logging.info('Creating configuration directory: ' + self.configdir)
            os.makedirs(self.configdir)

    def timestamp(self):
        return datetime.now().strftime('_%Y%b%d_%H%M%S')

    def filename(self, prefix='pyfab', suffix=None):
        return os.path.join(self.datadir,
                            prefix + self.timestamp() + suffix)

    def configname(self, object):
        classname = object.__class__.__name__
        return os.path.join(self.configdir, classname + '.json')

    def save(self, object):
        configuration = json.dumps(object.configuration(),
                                   indent=2,
                                   separators=(',', ': '),
                                   ensure_ascii=False)
        filename = self.configname(object)
        with io.open(filename, 'w', encoding='utf8') as configfile:
            configfile.write(unicode(configuration))

    def restore(self, object):
        try:
            filename = self.configname(object)
            config = json.load(io.open(filename))
            object.setConfiguration(config)
        except IOError as ex:
            msg = ('Could not read configuration file:\n\t' +
                   str(ex) +
                   '\n\tUsing default configuration.')
            logging.warning(msg)

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
