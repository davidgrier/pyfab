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
    system : object
        Spinnaker system
    cameras : object
        List of cameras attached to Spinnaker system
    camera : CameraPtr
        First camera on the list
    tldevice : INodeMap
        inodemap for the transport layer
    nodes : INodeMap
        Camera GenICam nodemap
    '''

    def __init__(self, **kwargs):
        super(QSpinnakerCamera, self).__init__(**kwargs)

        self.system = PySpin.System.GetInstance()
        self.cameras = self.system.GetCameras()
        if self.cameras.GetSize() < 1:
            logger.error('No Spinnaker cameras found')
        self.camera = self.cameras[0]
        self.camera.Init()
        self.nodes = self.camera.GetNodeMap()
        self.acquisitionmode = 'Continuous'
        self.camera.BeginAcquisition()

    def __del__(self):
        logger.debug('Cleaning up')
        self.camera.EndAcquisition()
        self.camera.DeInit()
        del self.camera
        self.cameras.Clear()
        self.system.ReleaseInstance()

    # Dynamically mapping to GenICam properties
    _pmap = {'width': 'Width',
             'height': 'Height',
             'x0': 'OffsetX',
             'y0': 'OffsetY',
             'acquisitionmode': 'AcquisitionMode',
             'exposure': 'ExposureTime',
             'framerate': 'AcquisitionFrameRate',
             'blacklevel': 'BlackLevel',
             'gain': 'Gain'}

    def _noattribute(self, name):
        msg = "'{0}' object has no attribute '{1}'"
        return msg.format(type(self).__name__, name)

    def __getattr__(self, name):
        try:
            sname = self._pmap[name]
            return self._getFValue(sname)
        except KeyError:
            raise AttributeError(self._noattribute(name))

    def __setattr__(self, name, value):
        try:
            pname = self._pmap[name]
            self._setFValue(pname, value)
        except KeyError:
            super(QSpinnakerCamera, self).__setattr__(name, value)

    @QtCore.pyqtSlot(object)
    def getProperty(self, name):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            return None

    @QtCore.pyqtSlot(object, object)
    def setProperty(self, name, value):
        if hasattr(self, name):
            setattr(self, name, value)

    def frame(self):
        res = self.camera.GetNextImage()
        if res.IsIncomplete():
            status = res.GetImageStatus()
            logger.warning('Incomplete Image: {}'.format(status))
        shape = (res.GetHeight(), res.GetWidth())
        return res.GetData().reshape(shape)

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
        node = self.nodes.GetNode(fname)
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
            logger.warning('Could not get {}'.format(fname))
        return value

    def _setFValue(self, fname, value):
        feature = self._feature(fname)
        if self._isEnum(feature) or self._isCommand(feature):
            feature.FromString(value)
        elif self._isWritable(feature):
            feature.SetValue(value)
        else:
            logger.warning('Could not set {} to {}'.format(fname, value))

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
        root = PySpin.CCategoryPtr(self.nodes.GetNode('Root'))
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
    import json

    cam = QSpinnakerCamera()
    print(cam.width)
    # print(json.dumps(cam.cameraInfo(), sort_keys=True, indent=4))
    # print(cam.framerate, cam.exposure)
    # print(cam.blacklevel, cam.gain)
    img = cam.frame()
    print(img.shape)
    del cam
