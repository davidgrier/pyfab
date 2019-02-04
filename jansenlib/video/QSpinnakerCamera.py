#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyqtgraph.Qt import QtCore
import PySpin

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QSpinnakerCamera(QtCore.QObject):

    '''Abstraction of FLiR camera for PyFab/Jansen

    Properties
    ----------
    camera : CameraPtr
        camera device in Spinnaker system
    properties : list
        strings containing names of adjustable properties
        Each named property may be treated like a python property
        and also can be accessed with getProperty() and setProperty()

    Methods
    -------
    getProperty(name):
        Get named property
    setProperty(name, value):
        Set named property to value
    read(): numpy.ndarray
        Return next available video frame
    '''

    def __init__(self, **kwargs):
        super(QSpinnakerCamera, self).__init__(**kwargs)

        # Initialize Spinnaker and get list of cameras
        self._system = PySpin.System.GetInstance()
        self._cameras = self._system.GetCameras()
        if self._cameras.GetSize() < 1:
            logger.error('No Spinnaker cameras found')
        # Work with first attached camera.  This can be generalized
        self.camera = self._cameras[0]
        self.camera.Init()
        # Get list of inodes in camera device.
        # These provide access to camera properties
        self._nodes = self.camera.GetNodeMap()
        self.properties = self._pmap.keys
        # Start acquisition
        self.acquisitionmode = 'Continuous'
        self.exposureauto = 'Off'
        self.gainauto = 'Off'
        self.camera.BeginAcquisition()

    def __del__(self):
        logger.debug('Cleaning up')
        self.camera.EndAcquisition()
        self.camera.DeInit()
        del self.camera
        self._cameras.Clear()
        self._system.ReleaseInstance()

    # Dynamic mapping for GenICam properties
    _pmap = {'width': 'Width',
             'height': 'Height',
             'x0': 'OffsetX',
             'y0': 'OffsetY',
             'acquisitionmode': 'AcquisitionMode',
             'framerate': 'AcquisitionFrameRate',
             'pixelformat': 'PixelFormat',
             'blacklevel': 'BlackLevel',
             'exposure': 'ExposureTime',
             'gain': 'Gain',
             'exposureauto': 'ExposureAuto',     # 'Off', 'Once', 'Continuous'
             'gainauto': 'GainAuto',
             'mirrored': 'ReverseX',
             'flipped': 'ReverseY'}

    def _noattribute(self, name):
        msg = "'{0}' object has no attribute '{1}'"
        return msg.format(type(self).__name__, name)

    def __getattr__(self, name):
        try:
            sname = self._pmap[name]
            return self._getFValue(sname)
        except KeyError:
            super(QSpinnakerCamera, self).__getattr__(name)

    def __setattr__(self, name, value):
        try:
            pname = self._pmap[name]
            self._setFValue(pname, value)
        except KeyError:
            super(QSpinnakerCamera, self).__setattr__(name, value)

    def getProperty(self, name):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            return None

    def setProperty(self, name, value):
        if hasattr(self, name):
            setattr(self, name, value)

    @property
    def gray(self):
        return self.pixelformat == 'Mono8'

    @gray.setter
    def gray(self, state):
        if (state):
            self.pixelformat = 'Mono8'
        else:
            self.pixelformat = 'RGB8'

    def read(self):
        res = self.camera.GetNextImage()
        error = res.IsIncomplete()
        if error:
            status = res.GetImageStatus()
            error_msg = res.GetImageStatusDescription(status)
            logger.warning('Incomplete Image: ' + error_msg)
        shape = (res.GetHeight(), res.GetWidth())
        return not error, res.GetData().reshape(shape)

    #
    # private methods for handling interactions with GenICam
    #
    _fmap = {PySpin.intfICategory: PySpin.CCategoryPtr,
             PySpin.intfIString: PySpin.CStringPtr,
             PySpin.intfIInteger: PySpin.CIntegerPtr,
             PySpin.intfIFloat: PySpin.CFloatPtr,
             PySpin.intfIBoolean: PySpin.CBooleanPtr,
             PySpin.intfICommand: PySpin.CCommandPtr,
             PySpin.intfIEnumeration: PySpin.CEnumerationPtr}

    def _feature(self, fname):
        node = self._nodes.GetNode(fname)
        type = node.GetPrincipalInterfaceType()
        feature = self._fmap[type](node)
        return feature

    def _getFValue(self, fname):
        feature = self._feature(fname)
        value = None
        if self._isEnum(feature) or self._isCommand(feature):
            value = feature.ToString()
        elif self._isReadable(feature):
            value = feature.GetValue()
        else:
            logger.warning('Could not access property: {}'.format(fname))
        return value

    def _setFValue(self, fname, value):
        feature = self._feature(fname)
        if not self._isWritable(feature):
            logger.warning('Property {} is not writable'.format(fname))
            return
        if self._isEnum(feature) or self._isCommand(feature):
            feature.FromString(value)
        else:
            feature.SetValue(value)

    def _isReadable(self, feature):
        return PySpin.IsAvailable(feature) and PySpin.IsReadable(feature)

    def _isWritable(self, feature):
        return PySpin.IsAvailable(feature) and PySpin.IsWritable(feature)

    def _isType(self, feature, typevalue):
        return (self._isReadable(feature) and
                feature.GetPrincipalInterfaceType() == typevalue)

    def _isCategory(self, feature):
        return self._isType(feature, PySpin.intfICategory)

    def _isEnum(self, feature):
        return self._isType(feature, PySpin.intfIEnumeration)

    def _isCommand(self, feature):
        return self._isType(feature, PySpin.intfICommand)

    #
    # Methods for introspection
    #
    def cameraInfo(self):
        root = PySpin.CCategoryPtr(self._nodes.GetNode('Root'))
        categories = dict()
        for category in root.GetFeatures():
            if self._isCategory(category):
                cname = category.GetName()
                cnode = self._feature(cname)
                features = dict()
                for node in cnode.GetFeatures():
                    if not self._isReadable(node):
                        continue
                    fname = node.GetName()
                    features[fname] = self._getFValue(fname)
                categories[cname] = features
        return categories

    def TLDeviceInfo(self):
        nodemap = self.camera.GetTLDeviceNodeMap()  # Transport layer
        try:
            info = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))
            if self._isReadable(info):
                features = info.GetFeatures()
                for feature in features:
                    this = PySpin.CValuePtr(feature)
                    print('{}: {}'.format(this.GetName(), this.ToString()))
            else:
                print('Device control information not available')
        except PySpin.SpinnakerException as ex:
            logger.warning('{}'.format(ex))


if __name__ == '__main__':
    cam = QSpinnakerCamera()
    print(cam.width)
    _, img = cam.read()
    print(img.shape)
    del cam
