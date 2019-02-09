# -*- coding: utf-8 -*-

from PyQt5.QtCore import (pyqtSlot, pyqtProperty)
from PyQt5.QtWidgets import (QWidget, QComboBox, QSpinBox,
                             QDoubleSpinBox, QCheckBox, QPushButton)
import inspect

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QSettingsWidget(QWidget):

    '''A glue class that connects a device with a GUI

    An object of this class associates properties of a device
    such as a camera with widgets in a GUI that are
    intended to control those properties.  Getting and setting
    the properties with this class ensures that both the device
    and the GUI are up to date and synchronized.

    ...

    Parameters
    ----------
    device : object
        Properties of this device will be selected
    ui : object
        UI object created with QtDesigner
    parent : QWidget
        Parent widget in PyQt hierarchy

    Attributes
    ----------
    properties : list
        List of properties that are managed by the object
    settings : dict
        Dictionary of properties and their settings

    Methods
    -------
    set(property, value)
        Set the named property to the specified value.
        This updates both the UI and the device.
    get(property) : int|float|bool
        Get the named property
    '''

    def __init__(self, parent=None, device=None, ui=None):
        '''
        Parameters
        ----------
        parent : QWidget
            Parent widget in PyQt hierarchy
        device : object
            Abstraction of device to be controlled
        ui : object
            GUI representing device created by QtDesigner
        '''
        super(QSettingsWidget, self).__init__(parent)
        self.ui = ui
        self.ui.setupUi(self)
        self._properties = []
        self.device = device

    @pyqtProperty(object)
    def device(self):
        '''Object representation of device to be controlled'''
        return self._device

    @device.setter
    def device(self, device):
        if device is None:
            self.setEnabled(False)
            self._device = None
            return
        if hasattr(self, 'device'):
            self.disconnectSignals()
        self._device = device
        self.getProperties()
        self.configureUi()
        self.updateUi()
        self.connectSignals()
        self.setEnabled(True)
        logger.info('device connected')

    def set(self, name, value):
        '''Set parameter on both device and UI

        Parameters
        ----------
        name : str
            Name of property
        value : scalar
            Value of property
        '''
        if name in self.properties:
            self.setDeviceProperty(name, value)
            self.setUiProperty(name, value)
        else:
            logger.warning('unknown property: {}'.format(name))

    def get(self, name):
        '''Return value from device (and UI)

        Parameters
        ----------
        name : str
            Name of property

        Returns
        -------
        value : scalar
            Value of property
        '''
        if name in self.properties:
            return getattr(self.device, name)
        else:
            logger.warning('unknown property: {}'.format(name))

    def waitForDevice(self):
        '''Wait until device is done processing last instruction'''
        if hasattr(self.device, 'busy'):
            print('got here')
            while self.device.busy():
                if self.device.error:
                    logger.warn('device error')

    def setDeviceProperty(self, name, value):
        '''Set device property and wait for operation to complete

        Parameters
        ----------
        name : str
            Name of property to set
        value : scalar
            Value to set
        '''
        if hasattr(self.device, name):
            setattr(self.device, name, value)
            logger.info('Setting {}: {}'.format(name, value))
            self.waitForDevice()

    def setUiProperty(self, name, value):
        '''Set UI property

        Parameters
        ----------
        name : str
            Name of property to set
        value : scalar
            Value to set
        '''
        wid = getattr(self.ui, name)
        if isinstance(wid, QDoubleSpinBox):
            wid.setValue(value)
        elif isinstance(wid, QSpinBox):
            wid.setValue(value)
        elif isinstance(wid, QComboBox):
            wid.setCurrentIndex(value)
        elif isinstance(wid, QCheckBox):
            if wid.isTristate():
                wid.setCheckState(value)
            else:
                wid.setCheckState(2*value)
        elif isinstance(wid, QPushButton):
            pass
        else:
            logger.warn('Unknown property: {}: {}'.format(name, type(wid)))

    @pyqtProperty(dict)
    def settings(self):
        '''Dictionary of properties and their values'''
        values = dict()
        for prop in self.properties:
            value = getattr(self.device, prop)
            if not inspect.ismethod(value):
                values[prop] = value
        return values

    @settings.setter
    def settings(self, values):
        for name in values:
            self.setDeviceProperty(name, values[name])
        self.updateUi()

    @pyqtProperty(list)
    def properties(self):
        '''List of properties managed by this object'''
        return self._properties

    def getProperties(self):
        '''Create list of properties

        Valid properties appear in both the device and the ui.
        Do not include private properties denoted with an underscore.
        '''
        if hasattr(self.device, 'properties'):
            dprops = self.device.properties
        else:
            dprops = [name for name, _ in inspect.getmembers(self.device)]
        uprops = [name for name, _ in inspect.getmembers(self.ui)]
        props = [name for name in dprops if name in uprops]
        self._properties = [name for name in props if '_' not in name]
        logger.debug(self._properties)

    def configureUi(self):
        logger.debug('configureUi should be overridden')

    @pyqtSlot()
    def updateUi(self):
        '''Update widgets with current values from device'''
        for prop in self.properties:
            val = getattr(self.device, prop)
            self.setUiProperty(prop, val)

    @pyqtSlot(int)
    @pyqtSlot(float)
    def updateDevice(self, value):
        '''Update device property when UI property is updated

        Connecting this slot to the appropriate UI signal ensures
        that device properties are updated when the user interacts
        with the UI.
        '''
        name = str(self.sender().objectName())
        self.setDeviceProperty(name, value)

    @pyqtSlot(bool)
    def autoUpdateDevice(self, flag):
        logger.debug('autoUpdateDevice')
        autosetproperty = self.sender.objectName()
        autosetmethod = getattr(self.device, autosetproperty)
        autosetmethod()
        self.waitForDevice()
        self.updateUi

    def connectSignals(self):
        for prop in self.properties:
            wid = getattr(self.ui, prop)
            if isinstance(wid, QDoubleSpinBox):
                wid.valueChanged.connect(self.updateDevice)
            elif isinstance(wid, QSpinBox):
                wid.valueChanged.connect(self.updateDevice)
            elif isinstance(wid, QComboBox):
                wid.currentIndexChanged.connect(self.updateDevice)
            elif isinstance(wid, QCheckBox):
                wid.stateChanged.connect(self.updateDevice)
            elif isinstance(wid, QPushButton):
                wid.clicked.connect(self.autoUpdateDevice)
            else:
                logger.warn('Unknown property: {}: {}'.format(prop, type(wid)))

    def disconnectSignals(self):
        for prop in self.properties:
            wid = getattr(self.ui, prop)
            if isinstance(wid, QDoubleSpinBox):
                wid.valueChanged.disconnect(self.updateDevice)
            elif isinstance(wid, QSpinBox):
                wid.valueChanged.disconnect(self.updateDevice)
            elif isinstance(wid, QComboBox):
                wid.currentIndexChanged.disconnect(self.updateDevice)
            elif isinstance(wid, QCheckBox):
                wid.stateChanged.disconnect(self.updateDevice)
            elif isinstance(wid, QPushButton):
                wid.clicked.disconnect(self.autoUpdateDevice)
            else:
                logger.warn('Unknown property: {}: {}'.format(prop, type(wid)))
