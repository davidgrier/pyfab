# -*- coding: utf-8 -*-

from PyQt5.QtCore import (pyqtSlot, pyqtProperty, QTimer)
from PyQt5.QtWidgets import (QFrame, QComboBox, QSpinBox,
                             QDoubleSpinBox, QCheckBox, QRadioButton,
                             QPushButton)
from .QSettingsWidget import QSettingsWidget


import inspect

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARNING)
logger.setLevel(logging.DEBUG)


class QMultiSettingsWidget(QSettingsWidget):

    ''' 
    QSettingsWidget that glues multiple devices to their respective widgets in the same GUI. Useful if multiple devices related devices have configurable properties in the same UI (i.e. see QVision. Note that SRCskip is in groupBox(es) of doVision, etc)
      
    Additional Parameters
    ----------
    devices: dict (prefix -> device)
        Dictionary storing secondary devices. Keys are a string 'prefix' appended to the respective GUI widget names
        example: devices['OTHER'] = myotherdevice --> widget 'OTHERproperty' is glued to myotherdevice.property
        properties which don't have any prefix correspond to 'device' as usual
      

    Additional Methods
    -------
    whichDevice(name)
        Reads device key from prefix of name, and returns corresponding property name and device
    '''

    def __init__(self, parent=None, device=None, ui=None, include=[], devices={}):
        self.devices = devices
        super(QMultiSettingsWidget, self).__init__(parent=parent, device=device, ui=ui, include=include)
        
        
    def whichDevice(self, name):
        for key in self.devices.keys():
            s = name.split(key)
            if s[0]=='' and name in self.propertiesOf(key):
                return s[1], self.devices[key]   ## If prefix matches dict key, return property name and device
        return name, self.device                           ## If no prefix matches, return original name and main device

    def _setDeviceProperty(self, name, value):
        logger.debug('Setting device: {}: {}'.format(name, value))
        propname, device = self.whichDevice(name)      
        if hasattr(device, propname):
            setattr(device, propname, value)
            self.waitForDevice(device)
            logger.info('Setting {}: {}'.format(name, value))
    
    def _getDeviceProperty(self, name):
        name, device = self.whichDevice(name)
        return getattr(device, name)
        
    def waitForDevice(self, device):
        '''Should be overridden by subclass'''
        pass

    def connectSignals(self, props=None, key=None):
        props = self.properties if key is None else self.propertiesOf(key)
#        print(props)
        super(QMultiSettingsWidget, self).connectSignals(props=props)
        
    def disconnectSignals(self, props=None, key=None):
        props = self.properties if key is None else self.propertiesOf(key)
#        print(props)
        super(QMultiSettingsWidget, self).disconnectSignals(props=props)
    
    @pyqtProperty(list)
    def properties(self):
        '''List of properties managed by this object'''
        props = []
        for key in list(self.devices.keys()):
            props.extend(self.propertiesOf(key))
        return props
    
    def propertiesOf(self, key='MAIN'):
        return self._properties[key] if key in self.devices.keys() else []
    
    def settingsOf(self, key='MAIN'):
        '''Dictionary of properties and their values'''
        values = dict()
        for prop in self.propertiesOf(key):
            value = self._getDeviceProperty(prop)
            if not inspect.ismethod(value):
                values[prop] = value
        return values   
    
    @pyqtProperty(object)
    def device(self):
        '''Object representation of device to be controlled'''
        return self.devices['MAIN'] if 'MAIN' in self.devices.keys() else None

    @device.setter
    def device(self, device):
        self.setDevice('MAIN', device)
        
    def setDevice(self, key, device):
        logger.info('Adding device {}'.format(key))
        if key not in self.devices.keys():
            logger.info('Adding new device {}'.format(key))
            if len(self._properties)==0:
                self._properties = {}
        else:
            self.disconnectSignals(key=key)
        self._properties[key] = []
        self.devices[key] = device
        if device is None:
            logger.info('Deleting NoneType object {}'.format(key))
            del self._properties[key]
            del self.devices[key]
            if len(self.devices)==0:
                self.setEnabled(False)
            return
        self.getProperties(key)
        self.configureUi()
        self.updateUi()
        self.connectSignals(key=key)
        self.setEnabled(True)
        logger.info('device {} connected'.format(key))
            
    def getProperties(self, key='MAIN'):
        '''Create dict of properties

        Valid properties appear in both the device and the ui,
        excluding private properties denoted with an underscore.
        '''
        device = self.devices[key]
        logger.debug('Getting Properties of {}'.format(key))
        dprops = [name for name, _ in inspect.getmembers(device)]
#         if key == 'MAIN':
        if key != 'MAIN':
            dprops = [key+dprop for dprop in dprops]
            
            
#         else:    
#             dprops = [key+dprop for dprop in dprops]
#             for prop in dprops:
#                 if prop in self._properties['MAIN']:
#                     self._properties['MAIN'].remove(prop)
        logger.debug('Device Properties: {}'.format(dprops))
        uprops = [name for name, _ in inspect.getmembers(self.ui)]
        logger.debug('UI Properties: {}'.format(uprops))
        props = [name for name in dprops if name in uprops]
        self._properties[key] = [name for name in props if ('_' not in name or name in self.include)]
        logger.debug('Common Properties: {}'.format(self._properties))
        
        
        
    def getAllProperties(self):
        for key in self.devices.keys:
            self.getProperties(key)
