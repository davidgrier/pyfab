# -*- coding: utf-8 -*-

import json
import os
import io
import platform
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)


class Configuration(object):

    """Save and restore configuration of objects for pyfab/jansen.

    The configuration object also includes utility functions for
    standard timestamps and standard file names."""

    def __init__(self, parent):
        self.parent = parent
        self.classname = parent.__class__.__name__.lower()
        self.datadir = os.path.expanduser('~/data/')
        self.configdir = os.path.expanduser('~/.{}/'.format(self.classname))
        if not os.path.exists(self.datadir):
            logger.info('Creating data directory: {}'.format(self.datadir))
            os.makedirs(self.datadir)
        if not os.path.exists(self.configdir):
            logger.info(
                'Creating configuration directory: {}'.format(self.configdir))
            os.makedirs(self.configdir)

    def timestamp(self):
        return datetime.now().strftime('_%Y%b%d_%H%M%S')

    def filename(self, prefix=None, suffix=None):
        if prefix is None:
            prefix = self.classname
        return os.path.join(self.datadir,
                            prefix + self.timestamp() + suffix)

    def configname(self, object):
        """Configuration file is named for the class of the object."""
        classname = object.__class__.__name__
        return os.path.join(self.configdir, classname + '.json')

    def save(self, object):
        """Save configuration of object as json file."""
        settings = json.dumps(object.settings,
                              indent=2,
                              separators=(',', ': '),
                              ensure_ascii=False)
        filename = self.configname(object)
        with io.open(filename, 'w', encoding='utf8') as configfile:
            if platform.python_version().startswith('3.'):
                configfile.write(str(settings))
            else:
                configfile.write(unicode(settings))

    def restore(self, object):
        """Restore object's configuration from json file."""
        try:
            filename = self.configname(object)
            settings = json.load(io.open(filename))
            object.settings = settings
        except IOError as ex:
            msg = ('Could not read configuration file:\n\t' +
                   str(ex) +
                   '\n\tUsing default configuration.')
            logger.warning(msg)

    def query_save(self, object):
        query = 'Save current configuration?'
        reply = QMessageBox.question(self.parent,
                                     'Confirmation',
                                     query,
                                     QMessageBox.Yes,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.save(object)
        else:
            pass
