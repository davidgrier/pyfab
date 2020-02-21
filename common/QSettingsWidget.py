# -*- coding: utf-8 -*-

from PyQt5.QtCore import (pyqtSlot, pyqtProperty, QTimer)
from PyQt5.QtWidgets import (QFrame, QComboBox, QSpinBox,
                             QDoubleSpinBox, QCheckBox, QRadioButton,
                             QPushButton)
import inspect

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
#logger.setLevel(logging.DEBUG)


class QSettingsWidget(QFrame):

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
        self.ui.closeEvent = self.closeEvent
        self._properties = []
        self.device = device

    def closeEvent(self):
        '''catch closeEvent to shut down device, if needed'''
        logger.debug('closeEvent should be overridden')

    def configureUi(self):
        '''Special-purpose widget settings

        Called when device is set. Used to handle
        device-specific configuration of UI elements,
        including setting limits on parameter widgets.
        '''
        logger.debug('configureUi should be overridden')

    def set(self, name, value):
        '''Set parameter on both device and UI

        Parameters
        ----------
        name : str
            Name of property
        value : scalar
            Value of property
        '''
        if name in self._properties:
            self._setDeviceProperty(name, value)
            actual = getattr(self.device, name)
            self._setUiProperty(name, actual)
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
        if name in self._properties:
            return getattr(self.device, name)
        else:
            logger.warning('unknown property: {}'.format(name))

    def _setDeviceProperty(self, name, value):
        '''Set device property and wait for operation to complete

        Parameters
        ----------
        name : str
            Name of property to set
        value : scalar
            Value to set
        '''
        logger.debug('Setting device: {}: {}'.format(name, value))
        if hasattr(self.device, name):
            setattr(self.device, name, value)
            self.waitForDevice()
            logger.info('Setting {}: {}'.format(name, value))

    def waitForDevice(self):
        '''Should be overridden by subclass'''
        pass

    def _setUiProperty(self, name, value):
        '''Set UI property

        Parameters
        ----------
        name : str
            Name of property to set
        value : scalar
            Value to set
        '''
        logger.debug('Setting UI: {}: {}'.format(name, value))
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
                wid.setChecked(value)
        elif isinstance(wid, QRadioButton):
            wid.setChecked(value)
        elif isinstance(wid, QPushButton):
            pass
        else:
            logger.warn('Unknown property: {}: {}'.format(name, type(wid)))

    @pyqtProperty(list)
    def properties(self):
        '''List of properties managed by this object'''
        return self._properties

    @pyqtProperty(dict)
    def settings(self):
        '''Dictionary of properties and their values'''
        values = dict()
        for prop in self._properties:
            value = getattr(self.device, prop)
            if not inspect.ismethod(value):
                values[prop] = value
        return values

    @settings.setter
    def settings(self, values):
        for name, value in values.items():
            self.set(name, value)

    @pyqtSlot()
    def updateUi(self):
        '''Update widgets with current values from device'''
        for prop in self._properties:
            val = getattr(self.device, prop)
            self._setUiProperty(prop, val)

    @pyqtSlot(bool)
    @pyqtSlot(int)
    @pyqtSlot(float)
    def updateDevice(self, value):
        '''Update device property when UI property is updated

        Connecting this slot to the appropriate UI signal ensures
        that device properties are updated when the user interacts
        with the UI.
        '''
        name = str(self.sender().objectName())
        logger.debug('Updating: {}: {}'.format(name, value))
        self._setDeviceProperty(name, value)

    @pyqtSlot(bool)
    def autoUpdateDevice(self, flag):
        logger.debug('autoUpdateDevice')
        autosetproperty = self.sender().objectName()
        autosetmethod = getattr(self.device, autosetproperty)
        autosetmethod()
        QTimer.singleShot(1000, self.updateUi)
        # self.waitForDevice()
        # self.updateUi()

    @pyqtProperty(object)
    def device(self):
        '''Object representation of device to be controlled'''
        return self._device

    @device.setter
    def device(self, device):
        self.disconnectSignals()
        logger.debug('Setting device: {}'.format(device))
        self._properties = []
        self._device = device
        if device is None:
            self.setEnabled(False)
            return
        self.getProperties()
        self.configureUi()
        self.updateUi()
        self.connectSignals()
        self.setEnabled(True)
        logger.info('device connected')

    def connectSignals(self):
        for prop in self._properties:
            logger.debug('Connecting {}'.format(prop))
            wid = getattr(self.ui, prop)
            if isinstance(wid, QDoubleSpinBox):
                wid.valueChanged.connect(self.updateDevice)
            elif isinstance(wid, QSpinBox):
                wid.valueChanged.connect(self.updateDevice)
            elif isinstance(wid, QComboBox):
                wid.currentIndexChanged.connect(self.updateDevice)
            elif isinstance(wid, QCheckBox):
                wid.stateChanged.connect(self.updateDevice)
            elif isinstance(wid, QRadioButton):
                wid.toggled.connect(self.updateDevice)                
            elif isinstance(wid, QPushButton):
                wid.clicked.connect(self.autoUpdateDevice)
            else:
                logger.warn('Unknown property: {}: {}'.format(prop, type(wid)))

    def disconnectSignals(self):
        for prop in self._properties:
            wid = getattr(self.ui, prop)
            if isinstance(wid, QDoubleSpinBox):
                wid.valueChanged.disconnect(self.updateDevice)
            elif isinstance(wid, QSpinBox):
                wid.valueChanged.disconnect(self.updateDevice)
            elif isinstance(wid, QComboBox):
                wid.currentIndexChanged.disconnect(self.updateDevice)
            elif isinstance(wid, QCheckBox):
                wid.stateChanged.disconnect(self.updateDevice)
            elif isinstance(wid, QRadioButton):
                wid.stateChanged.connect(self.updateDevice)    
            elif isinstance(wid, QPushButton):
                wid.clicked.disconnect(self.autoUpdateDevice)
            else:
                logger.warn('Unknown property: {}: {}'.format(prop, type(wid)))

    def getProperties(self):
        '''Create list of properties

        Valid properties appear in both the device and the ui,
        excluding private properties denoted with an underscore.
        '''
        logger.debug('Getting Properties')
        dprops = [name for name, _ in inspect.getmembers(self.device)]
        logger.debug('Device Properties: {}'.format(dprops))
        uprops = [name for name, _ in inspect.getmembers(self.ui)]
        logger.debug('UI Properties: {}'.format(uprops))
        props = [name for name in dprops if name in uprops]
        self._properties = [name for name in props if '_' not in name]
        logger.debug('Common Properties: {}'.format(self._properties))
