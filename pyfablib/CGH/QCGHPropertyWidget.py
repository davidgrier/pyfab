from PyQt4 import QtCore
from common.QPropertySheet import QPropertySheet


class QCGHPropertyWidget(QPropertySheet):

    def __init__(self, cgh, cam):
        super(QCGHPropertyWidget, self).__init__(title='CGH Pipeline')
        self.cgh = cgh
        slm = cgh.slm
        register = self.registerProperty
        self.wxs = register('xs', cgh.rs.x(), 0, slm.width(), self._xs)
        self.wys = register('ys', cgh.rs.y(), 0, slm.height(), self._ys)
        self.wqpp = register('qpp', cgh.qpp, 0., 1., self._qpp)
        self.walpha = register('alpha', cgh.alpha, 0.1, 10., self._alpha)
        self.wxc = register('xc', cgh.rc.x(), 0, cam.width(), self._xc)
        self.wyc = register('yc', cgh.rc.y(), 0, cam.height(), self._yc)
        self.wzc = register('zc', cgh.rc.z(), -500, 500, self._zc)
        self.wtheta = register('theta', cgh.theta, -180, 180, self._theta)
        self.wz0 = register('z0', cgh.z0, 10, 1000, self._z0)

    @QtCore.pyqtSlot()
    def _xs(self):
        rs = self.cgh.rs
        rs.setX(self.wxs.value)
        self.cgh.rs = rs

    @property
    def xs(self):
        return self.cgh.rs.x()

    @xs.setter
    def xs(self, xs):
        self.wxs.value = xs
        self._xs()

    @QtCore.pyqtSlot()
    def _ys(self):
        rs = self.cgh.rs
        rs.setY(self.wys.value)
        self.cgh.rs = rs

    @property
    def ys(self):
        return self.cgh.rs.y()

    @ys.setter
    def ys(self, ys):
        self.wys.value = ys
        self._ys()

    @QtCore.pyqtSlot()
    def _qpp(self):
        self.cgh.qpp = self.wqpp.value

    @property
    def qpp(self):
        return self.cgh.qpp

    @qpp.setter
    def qpp(self, qpp):
        self.wqpp.value = qpp
        self._qpp()

    @QtCore.pyqtSlot()
    def _alpha(self):
        self.cgh.alpha = self.walpha.value

    @property
    def alpha(self):
        return self.cgh.alpha

    @alpha.setter
    def alpha(self, alpha):
        self.walpha.value = alpha
        self._alpha()

    @QtCore.pyqtSlot()
    def _xc(self):
        rc = self.cgh.rc
        rc.setX(self.wxc.value)
        self.cgh.rc = rc

    @property
    def xc(self):
        return self.cgh.rc.x()

    @xc.setter
    def xc(self, xc):
        self.wxc.value = xc
        self._xc()

    @QtCore.pyqtSlot()
    def _yc(self):
        rc = self.cgh.rc
        rc.setY(self.wyc.value)
        self.cgh.rc = rc

    @property
    def yc(self):
        return self.cgh.rc.y()

    @yc.setter
    def yc(self, yc):
        self.wyc.value = yc
        self._yc()

    @QtCore.pyqtSlot()
    def _zc(self):
        rc = self.cgh.rc
        rc.setZ(self.wzc.value)
        self.cgh.rc = rc

    @property
    def zc(self):
        return self.cgh.rc.z()

    @zc.setter
    def zc(self, zc):
        self.wzc.value = zc
        self._zc()

    @QtCore.pyqtSlot()
    def _theta(self):
        self.cgh.theta = self.wtheta.value

    @property
    def theta(self):
        return self.cgh.theta

    @theta.setter
    def theta(self, theta):
        self.wtheta.value = theta
        self._theta()

    @QtCore.pyqtSlot()
    def _z0(self):
        self.cgh.z0 = self.wz0.value

    @property
    def z0(self):
        return self.cgh.z0

    @z0.setter
    def z0(self, z0):
        self.wz0.value = z0
        self._z0()

    @property
    def calibration(self):
        return {'xc': self.xc,
                'yc': self.yc,
                'zc': self.zc,
                'xs': self.xs,
                'ys': self.ys,
                'qpp': self.qpp,
                'alpha': self.alpha,
                'theta': self.theta,
                'z0': self.z0}

    @calibration.setter
    def calibration(self, values):
        if not isinstance(values, dict):
            return
        for attribute, value in values.iteritems():
            try:
                setattr(self, attribute, value)
            except AttributeError:
                print('unknown attribute:', attribute)
