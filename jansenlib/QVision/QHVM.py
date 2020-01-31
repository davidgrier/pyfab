# -*- coding: utf-8 -*-

from PyQt5.QtCore import pyqtSlot
from .QVision import QVision

from pylorenzmie.processing import Frame
from pylorenzmie.theory import Instrument

from scipy.stats import gaussian_kde

import CNNLorenzMie as cnn
import numpy as np
import pyqtgraph as pg
import matplotlib.cm as cm
import os
import json

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

path = os.path.expanduser('/'.join(cnn.__file__.split('/')[:-1]))
keras_head_path = path+'/keras_models/predict_stamp_best'
keras_model_path = keras_head_path+'.h5'
keras_config_path = keras_head_path+'.json'
with open(keras_config_path, 'r') as f:
    kconfig = json.load(f)


class QHVM(QVision):

    def __init__(self, parent=None):
        super(QHVM, self).__init__(parent=parent)
        ins = kconfig['instrument']
        self.instrument = Instrument(wavelength=ins['wavelength'],
                                     n_m=ins['n_m'],
                                     magnification=ins['magnification'])
        self.video.instrument = self.instrument

        self.localizer = None
        self.estimator = None

        self.configurePlots()
        self.configureChildUi()

    def configurePlots(self):
        self.ui.plot1.setBackground('w')
        self.ui.plot1.getAxis('bottom').setPen(0.1)
        self.ui.plot1.getAxis('left').setPen(0.1)
        self.ui.plot1.showGrid(x=True, y=True)
        self.ui.plot1.setLabel('bottom', 'a_p [um]')
        self.ui.plot1.setLabel('left', 'n_p')
        self.ui.plot2.setBackground('w')
        self.ui.plot2.getAxis('bottom').setPen(0.1)
        self.ui.plot2.getAxis('left').setPen(0.1)
        self.ui.plot2.showGrid(x=True, y=True)
        self.ui.plot2.setLabel('bottom', 't (s)')
        self.ui.plot2.setLabel('left', 'z(t)')

    def configureChildUi(self):
        self.ui.bDetect.setEnabled(True)
        self.ui.bEstimate.setEnabled(True)
        self.ui.bRefine.setEnabled(True)

    @pyqtSlot()
    def plot(self):
        if self.estimate:
            a_p = []
            n_p = []
            framenumbers = []
            trajectories = []
            for frame in self.video.frames:
                for feature in frame.features:
                    a_p.append(feature.model.particle.a_p)
                    n_p.append(feature.model.particle.n_p)
            for trajectory in self.video.trajectories:
                z_p = []
                for feature in trajectory.features:
                    z_p.append(feature.model.particle.z_p)
                trajectories.append(z_p)
                framenumbers.append(trajectory.framenumbers)
            # Characterization plot
            data = np.vstack([a_p, n_p])
            pdf = gaussian_kde(data)(data)
            norm = pdf/pdf.max()
            rgbs = []
            for val in norm:
                rgbs.append(self.getRgb(cm.hot, val))
            pos = [{'pos': data[:, i],
                    'pen': pg.mkPen(rgbs[i])} for i in range(len(a_p))]
            scatter = pg.ScatterPlotItem()
            self.ui.plot1.addItem(scatter)
            scatter.setData(pos)
            # z(t) plot
            grayscale = np.linspace(0, 1, len(trajectories), endpoint=True)
            for j in range(len(trajectories)):
                z_p = trajectories[j]
                f = np.array(framenumbers[j])
                curve = pg.PlotCurveItem()
                self.ui.plot2.addItem(curve)
                curve.setPen(pg.mkPen(self.getRgb(cm.gist_rainbow,
                                                  grayscale[j])))
                curve.setData(x=f/self.video.fps, y=z_p)
        self.sigCleanup.emit()

    def draw(self, detections):
        rois = []
        for detection in detections:
            x, y, w, h = detection['bbox']
            roi = pg.RectROI([x-w//2, y-h//2], [w, h], pen=self.pen)
            self.jansen.screen.addOverlay(roi)
            rois.append(roi)
        return rois

    def inflate(self, image):
        if len(image.shape) == 2:
            shape = image.shape
            inflated = np.stack((image,)*3, axis=-1)
        else:
            shape = (image.shape[0], image.shape[1])
            inflated = image
        return inflated, shape

    def predict(self, images, framenumbers, post=False):
        features = [[]]
        detections = [[]]
        frames = []
        inflated = []
        shape = (0, 0)
        for image in images:
            im, shape = self.inflate(image)
            inflated.append(im)
        if self.detect:
            detections = self.localizer.predict(img_list=inflated)
            detections = cnn.filters.nodoubles(detections, tol=0)
            detections = cnn.filters.no_edges(detections, tol=0,
                                              image_shape=shape)
            if self.estimator is None:
                pxls = (201, 201)
            else:
                pxls = self.estimator.pixels
            result = cnn.crop_feature(img_list=inflated,
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
                if post:
                    if framenumbers[idx] == maxframe:
                        logger.info("Refine complete!".format(frame.framenumber,
                                                              maxframe))
            frames.append(frame)
        return frames, detections

    def init_pipeline(self):
        self.jansen.screen.source.blockSignals(True)
        if self.localizer is None:
            self.localizer = cnn.Localizer(configuration='tinyholo',
                                           weights='_500k')
        if self.estimate:
            if self.estimator is None:
                self.estimator = cnn.Estimator(model_path=keras_model_path,
                                               config_file=kconfig)
            self.instrument = self.estimator.instrument
        self.jansen.screen.source.blockSignals(False)

    def clear_pipeline(self):
        self.detect = False
        self.estimate = False
        self.refine = False
