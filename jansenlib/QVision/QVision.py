# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSlot
from .QVisionWidget import Ui_QVisionWidget

from CNNLorenzMie.Localizer import Localizer
from CNNLorenzMie.Estimator import Estimator
from CNNLorenzMie.crop_feature import crop_feature
from CNNLorenzMie.filters.nodoubles import nodoubles
from CNNLorenzMie.filters.no_edges import no_edges
from pylorenzmie.theory import Video, Frame, Instrument

import numpy as np
import pyqtgraph as pg
import json
import os

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

path = os.path.expanduser("~/python/CNNLorenzMie")  # MAKE MORE GENERAL
keras_head_path = path+'/keras_models/predict_stamp_best'
keras_model_path = keras_head_path+'.h5'
keras_config_path = keras_head_path+'.json'
with open(keras_config_path, 'r') as f:
    kconfig = json.load(f)


class QVision(QWidget):

    def __init__(self, parent=None, nskip=3):
        super(QVision, self).__init__(parent)
        self.ui = Ui_QVisionWidget()
        self.ui.setupUi(self)

        self.jansen = None

        ins = kconfig['instrument']
        self.instrument = Instrument(wavelength=ins['wavelength'],
                                     n_m=ins['n_m'],
                                     magnification=ins['magnification'])
        self.video = Video(instrument=self.instrument)
        self.filename = None
        self.frames = []
        self.framenumbers = []

        self.recording = False

        self.localizer = None
        self.estimator = None

        self.detect = False
        self.estimate = False
        self.refine = False
        self.nskip = 0
        self.link_tol = 20.
        self.counter = self.nskip
        self.real_time = True
        self.save_frames = False
        self.save_trajectories = False
        self.save_feature_data = False

        self.rois = None
        self.pen = pg.mkPen(color='b', width=5)

        self.configurePlot()
        self.configureUi()
        self.connectSignals()

    def connectSignals(self):
        self.ui.breal.toggled.connect(self.handleRealTime)
        self.ui.bpost.toggled.connect(self.handlePost)
        self.ui.checkFrames.clicked.connect(self.handleSaveFrames)
        self.ui.checkTrajectories.clicked.connect(self.handleSaveTrajectories)
        self.ui.checkFeatureData.clicked.connect(self.handleSaveFeatureData)
        self.ui.bDetect.clicked.connect(self.handleDetect)
        self.ui.bEstimate.clicked.connect(self.handleEstimate)
        self.ui.bRefine.clicked.connect(self.handleRefine)
        self.ui.skipBox.valueChanged.connect(self.handleSkip)
        self.ui.spinTol.valueChanged.connect(self.handleLink)

    def configureUi(self):
        self.ui.bDetect.setChecked(self.detect)
        self.ui.bEstimate.setChecked(self.estimate)
        self.ui.bRefine.setChecked(self.refine)
        self.ui.checkFrames.setChecked(self.save_frames)
        self.ui.checkTrajectories.setChecked(self.save_trajectories)
        self.ui.breal.setChecked(self.real_time)
        self.ui.skipBox.setProperty("value", self.nskip)
        self.ui.spinTol.setProperty("value", self.link_tol)

    def configurePlot(self):
        self.ui.plot.setBackground('w')
        self.ui.plot.getAxis('bottom').setPen(0.1)
        self.ui.plot.getAxis('left').setPen(0.1)
        self.ui.plot.showGrid(x=True, y=True)
        self.ui.plot.setLabel('bottom', 'a_p [um]')
        self.ui.plot.setLabel('left', 'n_p')

    @pyqtSlot(np.ndarray)
    def process(self, image):
        self.remove(self.rois)
        if self.counter == 0:
            self.counter = self.nskip
            i = self.jansen.dvr.framenumber
            inflated, shape = self.inflate(image)
            if self.real_time:
                frames, detections = self.predict([inflated],
                                                  [i], shape)
                frame = frames[0]
                self.rois = self.draw(detections[0])
                if self.jansen.dvr.is_recording():
                    self.recording = True
                    if len(frame.features) != 0:
                        self.video.add(frames)
            else:
                if self.jansen.dvr.is_recording():
                    self.recording = True
                    self.frames.append(inflated)
                    self.framenumbers.append(i)
        else:
            self.counter -= 1
            self.rois = None
        if self.recording and not self.jansen.dvr.is_recording():
            self.recording = False
            self.jansen.screen.source.blockSignals(True)
            self.jansen.screen.pauseSignals(True)
            self.post_process()
            self.cleanup()
            self.jansen.screen.source.blockSignals(False)
            self.jansen.screen.pauseSignals(False)

    @pyqtSlot(bool)
    def handleDetect(self, selected):
        if selected:
            self.detect = True
            self.estimate = False
            self.refine = False
            self.ui.bEstimate.setChecked(False)
            self.ui.bRefine.setChecked(False)
            self.init_pipeline()
        else:
            self.clear_pipeline()

    @pyqtSlot(bool)
    def handleEstimate(self, selected):
        if selected:
            self.detect = True
            self.estimate = True
            self.refine = False
            self.ui.bDetect.setChecked(False)
            self.ui.bRefine.setChecked(False)
            self.init_pipeline()
        else:
            self.clear_pipeline()

    @pyqtSlot(bool)
    def handleRefine(self, selected):
        if selected:
            self.detect = True
            self.estimate = True
            self.refine = True
            self.ui.bDetect.setChecked(False)
            self.ui.bEstimate.setChecked(False)
            self.init_pipeline()
        else:
            self.clear_pipeline()

    @pyqtSlot(bool)
    def handleRealTime(self, selected):
        self.real_time = selected

    @pyqtSlot(bool)
    def handlePost(self, selected):
        self.real_time = not selected

    @pyqtSlot(bool)
    def handleSaveFrames(self, selected):
        self.save_frames = selected

    @pyqtSlot(bool)
    def handleSaveTrajectories(self, selected):
        self.save_trajectories = selected

    @pyqtSlot(bool)
    def handleSaveFeatureData(self, selected):
        self.save_feature_data = selected

    @pyqtSlot(int)
    def handleSkip(self, nskip):
        self.nskip = nskip

    @pyqtSlot(float)
    def handleLink(self, tol):
        self.link_tol = tol

    def post_process(self):
        if not self.real_time:
            shape = self.frames[0].shape
            frames, detections = self.predict(self.frames,
                                              self.framenumbers,
                                              (shape[0], shape[1]),
                                              post=True)
            self.video.add(frames)
            self.frames = []
            self.framenumbers = []

    def cleanup(self):
        omit, omit_feat = ([], [])
        if not self.save_frames:
            omit.append('frames')
        if not self.save_trajectories:
            omit.append('trajectories')
        else:
            self.video.set_trajectories(search_range=self.link_tol,
                                        memory=int(self.nskip+3))
        if not self.save_feature_data:
            omit_feat.append('data')
        if self.save_frames or self.save_trajectories:
            self.filename = self.jansen.dvr.filename.split(".")[0] + '.json'
            self.video.serialize(filename=self.filename,
                                 omit=omit,
                                 omit_frame=['data'],
                                 omit_feat=omit_feat)
            logger.info("{} saved.".format(self.filename))
        self.video = Video(instrument=self.instrument)

    def inflate(self, image):
        if len(image.shape) == 2:
            shape = image.shape
            inflated = np.stack((image,)*3, axis=-1)
        else:
            shape = (image.shape[0], image.shape[1])
            inflated = image
        return inflated, shape

    def predict(self, images, framenumbers, shape, post=False):
        features = [[]]
        detections = [[]]
        frames = []
        if self.detect:
            detections = self.localizer.predict(img_list=images)
            detections = nodoubles(detections, tol=0)
            detections = no_edges(detections, tol=0,
                                  image_shape=shape)
            if self.estimator is None:
                pxls = (201, 201)
            else:
                pxls = self.estimator.pixels
            result = crop_feature(img_list=images,
                                  xy_preds=detections,
                                  new_shape=pxls)
            if post:
                logger.info("Detection complete!")
            features, est_images, scales = result
            if self.estimate:
                structure = list(map(len, features))
                char_predictions = self.estimator.predict(
                    img_list=est_images, scale_list=scales)
                zpop = char_predictions['z_p']
                apop = char_predictions['a_p']
                npop = char_predictions['n_p']
                for framenum in range(len(structure)):
                    listlen = structure[framenum]
                    frame = features[framenum]
                    index = 0
                    while listlen > index:
                        feature = frame[index]
                        feature.model.particle.z_p = zpop.pop(0)
                        feature.model.particle.a_p = apop.pop(0)
                        feature.model.particle.n_p = npop.pop(0)
                        feature.model.instrument = self.estimator.instrument
                        index += 1
                if post:
                    logger.info("Estimation complete!")
        maxframe = max(framenumbers)
        for idx, feat_list in enumerate(features):
            frame = Frame(features=feat_list,
                          framenumber=framenumbers[idx],
                          instrument=self.video.instrument)
            if self.refine:
                m = 'lm' if self.real_time else 'amoeba-lm'
                for feature in frame.features:
                    feature.model.double_precision = False
                    feature.lm_settings.options['max_nfev'] = 250
                    for f in self.jansen.screen.filters:
                        if 'samplehold' in str(f):
                            feature.data = feature.data / np.mean(feature.data)
                    result = feature.optimize(method=m)
                    print(result)
                if post:
                    if framenumbers[idx] == maxframe:
                        logger.info("Refine complete!".format(frame.framenumber,
                                                              maxframe))
            frames.append(frame)
        return frames, detections

    def draw(self, detections):
        rois = []
        for detection in detections:
            x, y, w, h = detection['bbox']
            roi = pg.RectROI([x-w//2, y-h//2], [w, h], pen=self.pen)
            self.jansen.screen.addOverlay(roi)
            rois.append(roi)
        return rois

    def remove(self, rois):
        if rois is not None:
            for rect in rois:
                self.jansen.screen.removeOverlay(rect)

    def init_pipeline(self):
        self.jansen.screen.source.blockSignals(True)
        if self.localizer is None:
            self.localizer = Localizer(configuration='tinyholo',
                                       weights='_500k')
        if self.estimate:
            if self.estimator is None:
                self.estimator = Estimator(model_path=keras_model_path,
                                           config_file=kconfig)
            self.instrument = self.estimator.instrument
        self.jansen.screen.source.blockSignals(False)

    def clear_pipeline(self):
        self.detect = False
        self.estimate = False
        self.refine = False
