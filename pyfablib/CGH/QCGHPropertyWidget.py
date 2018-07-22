# -*- coding: utf-8 -*-

"""Control panel for configuring hologram calculation."""

from common.QPropertySheet import QPropertySheet
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)


class QCGHPropertyWidget(QPropertySheet):

    def __init__(self, parent):
        super(QCGHPropertyWidget, self).__init__(title='CGH Pipeline')
        cgh = parent.cgh
        slm = cgh.slm
        cam = parent.screen.video.defaultSource
        register = self.registerProperty
        setter = cgh.setProperty

        header = 'CGH Calibration: '
        tip = header + 'Overall scale factor [mrad/pixel]'
        self.wqpp = register('qpp', cgh.qpp, 0.1, 100., setter, tip)
        tip = header + 'x-y anisotropy'
        self.walpha = register('alpha', cgh.alpha, 0.1, 10., setter, tip)
        tip = header + 'x coordinate of optical axis on camera [pixel]'
        self.wxc = register('xc', cgh.rc.x(), 0, cam.width, setter, tip)
        tip = header + 'y coordinate of optical axis on camera [pixel]'
        self.wyc = register('yc', cgh.rc.y(), 0, cam.height, setter, tip)
        tip = header + 'axial coordinate of focal plane [pixel]'
        self.wzc = register('zc', cgh.rc.z(), -500, 500, setter, tip)
        tip = header + 'camera orientation relative to SLM [degrees]'
        self.wthetac = register('thetac', cgh.thetac, -180, 180, setter, tip)
        tip = header + 'x coordinate of optical axis on SLM [pixel]'
        self.wxs = register('xs', cgh.rs.x(), 0, slm.width(), setter, tip)
        tip = header + 'y coordinate of optical axis on SLM [pixel]'
        self.wys = register('ys', cgh.rs.y(), 0, slm.height(), setter, tip)
        tip = header + 'Axial splay factor [radian/pixel]'
        self.wk0 = register('k0', cgh.k0, -10, 10, setter, tip)

    def configuration(self):
        return {'xs': self.wxs.value,
                'ys': self.wys.value,
                'qpp': self.wqpp.value,
                'alpha': self.walpha.value,
                'xc': self.wxc.value,
                'yc': self.wyc.value,
                'zc': self.wzc.value,
                'thetac': self.wthetac.value,
                'k0': self.wk0.value}

    def setConfiguration(self, properties):
        for property, value in properties.items():
            try:
                self.__dict__['w' + property].value = value
            except KeyError:
                logger.warning('Unknown attribute: {}'.format(property))
