#!/usr/bin/env python
# -*- coding: utf-8 -*-

import PySpin
import cv2

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

'''
Technical Reference:
http://softwareservices.flir.com/BFS-U3-123S6/latest/Model/public/index.html

NOTE:
USB 3.x communication on Ubuntu 16.04 through 20.04 requires
> sudo sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'

This can be set permanently by 
1. Editing /etc/default/grub
Change:
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
to:
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash usbcore.usbfs_memory_mb=1000"
2. > sudo update-grub
3. > sudo reboot now
'''


class SpinnakerCamera(object):

    '''Abstraction of FLiR camera for PyFab/Jansen

    ...

    Properties
    ==========
    device: PySpin.CameraPtr
        camera device in Spinnaker system
    cameraname : str
        Vendor and camera model

    Acquisition Control
    -------------------
    acquisitionmode : str
        'Continuous': acquire frames continuously (Default)
        'MultiFrame': acquire specified number of frames
        'SingleFrame': acquire one image before stopping
    acquisitionframecount : int
        Number of frames to acquire with MultiFrame
    exposuremode : str
        'Timed', 'TriggerWidth'
        Method for initiating exposure
        Default: 'Timed'
    exposuretime : float
        Exposure time in microseconds when exposuremode='Timed'
    exposuretimerange : (float, float)
        Range of exposure times in microseconds
    exposureauto: str
        Automatic exposure mode
        'Off', 'Once', 'Continuous'
        Default: 'Off'
    framerate : float
        Acquisition frame rate in Hertz
    framerateenable : bool
        Manually control framerate
    frameraterange : (float, float)
        Range of frame rates in Hertz
    framerateauto: str
        Enable automatic control of frame rate
        'Off', 'Continuous'
        Default: 'Off'

    NOTE: Trigger Commands are not yet implemented

    Analog Control
    --------------
    gain : float
        Amplification of video signal in dB
    gainrange : (float, float)
        Range of gain values in dB
    gainauto: 'Off', 'Once', 'Continuous'
        Enable automatic control of gain
        Default: 'Off'
    blacklevel : int
        Offset of video signal in camera-specific units
    blacklevelrange : (int, int)
        Range of black level values
    gamma : float
        Gamma correction of pixel intensity
    gammarange : (float, float)
        Range of gamma values
    gammaenable : bool
        Enable gamma correction
        Default: True
    sharpening : float
        Amount of sharpening to apply to image
        Default: 2.
    sharpeningrange : (float, float)
        Range of sharpening values
    sharpeningenable : bool
        Enable image sharpening
        Default: False
    sharpeningthreshold : float
        Only sharpen regions with intensity changes greater than threshold
    sharpeningthresholdrange : (float, float)
        Range of sharpening threshold values

    Image Format Control
    --------------------
    flipped : bool
        Vertically flip image
    mirrored : bool
        Horizontally flip image
    gray: bool
        read() returns single-channel (grayscale) image if True

    Methods
    =======
    open(index) :
        Open FLiR camera specified by index
        Default: index=0, first camera
    close() :
        Close camera
    start() :
        Start image acquisition
    stop() : 
        Stop image acquisition
    read() : (bool, numpy.ndarray)
        Return next available video frame
    '''

    def __init__(self,
                 blacklevelselector='All',
                 framerateenable=True,
                 gammaenable=True,
                 sharpeningenable=False,
                 acquisitionmode=None,
                 exposureauto=None,
                 exposuremode=None,
                 framerateauto=None,
                 gainauto=None,
                 gray=None,
                 flipped=None,
                 mirrored=None):
        self.open()

        # Enable access to controls
        self.blacklevelselector = blacklevelselector
        self.framerateenable = framerateenable
        self.gammaenable = gammaenable
        self.sharpeningenable = sharpeningenable

        # Start acquisition
        self.acquisitionmode = acquisitionmode or 'Continuous'
        self.exposureauto = exposureauto or 'Off'
        self.exposuremode = exposuremode or 'Timed'
        self.flipped = flipped or False
        self.framerateauto = framerateauto or 'Off'
        self.gainauto = gainauto or 'Off'
        self.gray = gray or True
        self.mirrored = mirrored or False
        self.sharpeningauto = False

        self.start()
        ready, frame = self.read()

    def __del__(self):
        self.close()

    def open(self, index=0):
        # Initialize Spinnaker and get list of cameras
        self._system = PySpin.System.GetInstance()
        self._devices = self._system.GetCameras()
        if self._devices.GetSize() < 1:
            raise IndexError('No Spinnaker cameras found')

        # Work with first attached camera.  This can be generalized
        self.device = self._devices[index]
        self.device.Init()
        # Camera inodes provide access to device properties
        self._nodes = self.device.GetNodeMap()

    def close(self):
        logger.debug('Cleaning up')
        self.stop()
        self.device.DeInit()
        del self.device
        self._devices.Clear()
        self._system.ReleaseInstance()

    def start(self):
        '''Start image acquisition'''
        self.device.BeginAcquisition()

    def stop(self):
        '''Stop image acquisition'''
        self.device.EndAcquisition()

    def read(self):
        '''The whole point of the thing: Gimme da piccy'''
        res = self.device.GetNextImage()
        error = res.IsIncomplete()
        if error:
            status = res.GetImageStatus()
            error_msg = res.GetImageStatusDescription(status)
            logger.warning('Incomplete Image: ' + error_msg)
            return not error, None
        shape = (res.GetHeight(), res.GetWidth())
        image = res.GetData().reshape(shape)
        if self.flipped:
            image = cv2.flip(image, 0)
        return not error, image

    @property
    def acquisitionframecount(self):
        return self._get_feature('AcquisitionFrameCount')

    @acquisitionframecount.setter
    def acquisitionframecount(self, value):
        self._set_feature('AcquisitionFrameCount', value)

    @property
    def acquisitionmode(self):
        return self._get_feature('AcquisitionMode')

    @acquisitionmode.setter
    def acquisitionmode(self, mode):
        self._set_feature('AcquisitionMode', mode)

    @property
    def blacklevel(self):
        return self._get_feature('BlackLevel')

    @blacklevel.setter
    def blacklevel(self, value):
        self._set_feature('BlackLevel', value)

    @property
    def blacklevelrange(self):
        return self._feature_range('BlackLevel')

    @property
    def blacklevelselector(self):
        return self._get_feature('BlackLevelSelector')

    @blacklevelselector.setter
    def blacklevelselector(self, value):
        self._set_feature('BlackLevelSelector', value)

    @property
    def cameraname(self):
        vendor = self._get_feature('DeviceVendorName')
        model = self._get_feature('DeviceModelName')
        return '{} {}'.format(vendor, model)

    @property
    def exposureauto(self):
        return self._get_feature('ExposureAuto')

    @exposureauto.setter
    def exposureauto(self, value):
        self._set_feature('ExposureAuto', value)

    @property
    def exposuremode(self):
        return self._get_feature('ExposureMode')

    @exposuremode.setter
    def exposuremode(self, value):
        self._set_feature('ExposureMode', value)

    @property
    def exposuretime(self):
        return self._get_feature('ExposureTime')

    @exposuretime.setter
    def exposuretime(self, value):
        self._set_feature('ExposureTime', value)

    @property
    def exposuretimerange(self):
        return self._feature_range('ExposureTime')

    @property
    def flipped(self):
        return self._get_feature('ReverseY')

    @flipped.setter
    def flipped(self, state):
        self._set_feature('ReverseY', bool(state))

    @property
    def framerate(self):
        return self._get_feature('AcquisitionFrameRate')

    @framerate.setter
    def framerate(self, value):
        self._set_feature('AcquisitionFrameRate', value)

    @property
    def framerateenable(self):
        return self._get_feature('AcquisitionFrameRateEnable')

    @framerateenable.setter
    def framerateenable(self, state):
        self._set_feature('AcquisitionFrameRateEnable', state)

    @property
    def frameraterange(self):
        return self._feature_range('AcquisitionFrameRate')

    @property
    def frameratemax(self):
        return self._feature('AcquisitionFrameRate').GetMax()

    @property
    def frameratemin(self):
        return self._feature('AcquisitionFrameRate').GetMin()

    @property
    def gain(self):
        return self._get_feature('Gain')

    @gain.setter
    def gain(self, value):
        self._set_feature('Gain', value)

    @property
    def gainauto(self):
        return self._get_feature('GainAuto')

    @gainauto.setter
    def gainauto(self, value):
        self._set_feature('GainAuto', value)

    @property
    def gainrange(self):
        return self._feature_range('Gain')

    @property
    def gamma(self):
        return self._get_feature('Gamma')

    @gamma.setter
    def gamma(self, value):
        self._set_feature('Gamma', value)

    @property
    def gammaenable(self):
        return self._get_feature('GammaEnable')

    @gammaenable.setter
    def gammaenable(self, state):
        self._set_feature('GammaEnable', bool(state))

    @property
    def gammarange(self):
        return self._feature_range('Gamma')

    @property
    def gray(self):
        return self.pixelformat == 'Mono8'

    @gray.setter
    def gray(self, state):
        if (state):
            self.pixelformat = 'Mono8'
        else:
            self.pixelformat = 'RGB8'

    @property
    def height(self):
        return self._get_feature('Height')

    @height.setter
    def height(self, value):
        self._set_feature('Height', value)

    @property
    def heightmax(self):
        return self._get_feature('HeightMax')

    @property
    def mirrored(self):
        return self._get_feature('ReverseX')

    @mirrored.setter
    def mirrored(self, state):
        self._set_feature('ReverseX', bool(state))

    @property
    def pixelformat(self):
        return self._get_feature('PixelFormat')

    @pixelformat.setter
    def pixelformat(self, value):
        self._set_feature('PixelFormat', value)

    @property
    def sharpening(self):
        return self._get_feature('Sharpening')

    @sharpening.setter
    def sharpening(self, value):
        self._set_feature('Sharpening', value)

    @property
    def sharpeningauto(self):
        return self._get_feature('SharpeningAuto')

    @sharpeningauto.setter
    def sharpeningauto(self, value):
        self._set_feature('SharpeningAuto', value)

    @property
    def sharpeningenable(self):
        return self._get_feature('SharpeningEnable')

    @sharpeningenable.setter
    def sharpeningenable(self, state):
        self._set_feature('SharpeningEnable', bool(state))

    @property
    def sharpeningrange(self):
        return (1., 8.)

    @property
    def sharpeningthreshold(self):
        return self._get_feature('SharpeningThreshold')

    @sharpeningthreshold.setter
    def sharpeningthreshold(self, value):
        self._set_feature('SharpeningThreshold', value)

    @property
    def sharpeningthresholdrange(self):
        return (0., 0.25)

    '''
    @property
    def videomode(self):
        return self._feature('VideoMode').GetValue()

    @videomode.setter
    def videomode(self, mode):
        self.stop()
        self._feature('VideoMode').SetValue(mode)
        self.start()
    '''

    @property
    def width(self):
        return self._get_feature('Width')

    @width.setter
    def width(self, value):
        self._set_feature('Width', value)

    @property
    def widthmax(self):
        return self._get_feature('WidthMax')

    def autoexposure(self):
        self.exposureauto = 'Once'

    def autogain(self):
        self.gainauto = 'Once'

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
        '''Return inode for named feature'''
        feature = None
        try:
            node = self._nodes.GetNode(fname)
            type = node.GetPrincipalInterfaceType()
            feature = self._fmap[type](node)
        except Exception as ex:
            logger.warn('Could not access Property: {} {}'.format(fname, ex))
        return feature

    def _get_feature(self, fname):
        value = None
        feature = self._feature(fname)
        if self._is_enum(feature) or self._is_command(feature):
            value = feature.ToString()
        elif self._is_category(feature):
            nodes = feature.GetFeatures()
            value = dict()
            for node in nodes:
                name = node.GetName()
                value[name] = self._get_feature(name)
        elif self._is_readable(feature):
            value = feature.GetValue()
        logger.debug('Getting {}: {}'.format(fname, value))
        return value

    def _set_feature(self, fname, value):
        logger.debug('Setting {}: {}'.format(fname, value))
        feature = self._feature(fname)
        if not self._is_writable(feature):
            logger.warning('Property {} is not writable'.format(fname))
            return
        try:
            if self._is_enum(feature) or self._is_command(feature):
                feature.FromString(value)
            else:
                feature.SetValue(value)
        except PySpin.SpinnakerException as ex:
            logger.warning('Could not set {}: {}'.format(fname, ex))

    def _feature_range(self, fname):
        '''Return minimum and maximum values of named feature'''
        feature = self._feature(fname)
        try:
            range = (feature.GetMin(), feature.GetMax())
        except PySpin.SpinnakerException as ex:
            logger.warning('Could not get range of {}: {}'.format(fname, ex))
            range = None
        return range

    def _is_readable(self, feature):
        return PySpin.IsAvailable(feature) and PySpin.IsReadable(feature)

    def _is_writable(self, feature):
        return PySpin.IsAvailable(feature) and PySpin.IsWritable(feature)

    def _is_type(self, feature, typevalue):
        return (self._is_readable(feature) and
                feature.GetPrincipalInterfaceType() == typevalue)

    def _is_category(self, feature):
        return self._is_type(feature, PySpin.intfICategory)

    def _is_enum(self, feature):
        return self._is_type(feature, PySpin.intfIEnumeration)

    def _is_command(self, feature):
        return self._is_type(feature, PySpin.intfICommand)

    #
    # Methods for introspection
    #
    def camera_info(self):
        '''Return dict of camera inodes and values'''
        return self._get_feature('Root')

    def transport_info(self):
        '''Return dict of Transport Layer Device inodes and values'''
        nodemap = self.device.GetTLDeviceNodeMap()  # Transport layer
        try:
            info = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))
            if self._is_readable(info):
                features = info.GetFeatures()
                for feature in features:
                    this = PySpin.CValuePtr(feature)
                    print('{}: {}'.format(this.GetName(), this.ToString()))
                else:
                    print('Device control information not available')
        except PySpin.SpinnakerException as ex:
            logger.warning('{}'.format(ex))


if __name__ == '__main__':
    cam = SpinnakerCamera()
    print(cam.width)
    _, img = cam.read()
    print(img.shape)
    del cam
