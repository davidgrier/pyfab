from PyQt4 import QtCore
from common.QPropertySheet import QPropertySheet


class QCGHPropertyWidget(QPropertySheet):

    def __init__(self, cgh, camera):
        super(QCGHPropertyWidget, self).__init__(title='CGH Pipeline')
        self.cgh = cgh
        self.wxs = self.registerProperty('xs', cgh.rs.x(), 0, cgh.slm.width())
        self.wys = self.registerProperty('ys', cgh.rs.y(), 0, cgh.slm.height())
        self.wqpp = self.registerProperty('qpp', cgh.qpp, 0., 1.)
        self.walpha = self.registerProperty('alpha', cgh.alpha, 0.1, 10.)
        self.wxc = self.registerProperty('xc', cgh.rc.x(), 0, camera.width())
        self.wyc = self.registerProperty('yc', cgh.rc.y(), 0, camera.height())
        self.wzc = self.registerProperty('zc', cgh.rc.z(), -500, 500)
        self.wtheta = self.registerProperty('theta', cgh.theta, -180, 180)
        self.wz0 = self.registerProperty('z0', cgh.z0, 10, 1000)
        self.wxs.valueChanged.connect(self.updateXs)
        self.wys.valueChanged.connect(self.updateYs)
        self.wqpp.valueChanged.connect(self.updateQpp)
        self.walpha.valueChanged.connect(self.updateAlpha)
        self.wxc.valueChanged.connect(self.updateXc)
        self.wyc.valueChanged.connect(self.updateYc)
        self.wzc.valueChanged.connect(self.updateZc)
        self.wtheta.valueChanged.connect(self.updateTheta)
        self.wz0.valueChanged.connect(self.updateZ0)

    @QtCore.pyqtSlot()
    def updateXs(self):
        rs = self.cgh.rs
        rs.setX(self.wxs.value)
        self.cgh.rs = rs

    @property
    def xs(self):
        return self.cgh.rs.x()

    @xs.setter
    def xs(self, xs):
        self.wxs.value = xs
        self.updateXs()

    @QtCore.pyqtSlot()
    def updateYs(self):
        rs = self.cgh.rs
        rs.setY(self.wys.value)
        self.cgh.rs = rs

    @property
    def ys(self):
        return self.cgh.rs.y()

    @ys.setter
    def ys(self, ys):
        self.wys.value = ys
        self.updateYs()

    @QtCore.pyqtSlot()
    def updateQpp(self):
        self.cgh.qpp = self.wqpp.value

    @property
    def qpp(self):
        return self.cgh.qpp

    @qpp.setter
    def qpp(self, qpp):
        self.wqpp.value = qpp
        self.updateQpp()

    @QtCore.pyqtSlot()
    def updateAlpha(self):
        self.cgh.alpha = self.walpha.value

    @property
    def alpha(self):
        return self.cgh.alpha

    @alpha.setter
    def alpha(self, alpha):
        self.walpha.value = alpha
        self.updateAlpha()

    @QtCore.pyqtSlot()
    def updateXc(self):
        rc = self.cgh.rc
        rc.setX(self.wxc.value)
        self.cgh.rc = rc

    @property
    def xc(self):
        return self.cgh.rc.x()

    @xc.setter
    def xc(self, xc):
        self.wxc.value = xc
        self.updateXc()

    @QtCore.pyqtSlot()
    def updateYc(self):
        rc = self.cgh.rc
        rc.setY(self.wyc.value)
        self.cgh.rc = rc

    @property
    def yc(self):
        return self.cgh.rc.y()

    @yc.setter
    def yc(self, yc):
        self.wyc.value = yc
        self.updateYc()

    @QtCore.pyqtSlot()
    def updateZc(self):
        rc = self.cgh.rc
        rc.setZ(self.wzc.value)
        self.cgh.rc = rc

    @property
    def zc(self):
        return self.cgh.rc.z()

    @zc.setter
    def zc(self, zc):
        self.wzc.value = zc
        self.updateZc()

    @QtCore.pyqtSlot()
    def updateTheta(self):
        self.cgh.theta = self.wtheta.value

    @property
    def theta(self):
        return self.cgh.theta

    @theta.setter
    def theta(self, theta):
        self.wtheta.value = theta
        self.updateTheta()

    @QtCore.pyqtSlot()
    def updateZ0(self):
        self.cgh.z0 = self.wz0.value

    @property
    def z0(self):
        return self.cgh.z0

    @z0.setter
    def z0(self, z0):
        self.wz0.value = z0
        self.updateZ0()

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
