from QPropertySheet import QPropertySheet


class QCGH(QPropertySheet):

    def __init__(self, cgh=None):
        super(QCGH, self).__init__(title='CGH Pipeline')
        self.cgh = cgh
        self.wxs = self.registerProperty('xs', cgh.rs.x(), 0, cgh.slm.width())
        self.wys = self.registerProperty('ys', cgh.rs.y(), 0, cgh.slm.height())
        self.wqpp = self.registerProperty('qpp', cgh.qpp, 0., 1.)
        self.walpha = self.registerProperty('alpha', cgh.alpha, 0.1, 10.)
        self.wxc = self.registerProperty('xc', cgh.rc.x(), 0, 512)
        self.wyc = self.registerProperty('yc', cgh.rc.y(), 0, 512)
        self.wtheta = self.registerProperty('theta', cgh.theta, -1, 1)
        self.wxs.valueChanged.connect(self.updateXs)
        self.wys.valueChanged.connect(self.updateYs)
        self.walpha.valueChanged.connect(self.updateAlpha)
        self.wxc.valueChanged.connect(self.updateXc)
        self.wyc.valueChanged.connect(self.updateYc)
        self.wtheta.valueChanged.connect(self.updateTheta)

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

    def updateAlpha(self):
        self.cgh.alpha = self.walpha.value

    @property
    def alpha(self):
        return self.cgh.alpha

    @alpha.setter
    def alpha(self, alpha):
        self.walpha.value = alpha
        self.updateAlpha()

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

    @property
    def theta(self):
        return self.cgh.theta

    def updateTheta(self):
        self.cgh.theta = self.wtheta.value

    @theta.setter
    def theta(self, theta):
        self.wtheta.value = theta
        self.updateTheta()


def main():
    from PyQt4 import QtGui
    from QSLM import QSLM
    from CGH import CGH
    import sys

    app = QtGui.QApplication(sys.argv)
    slm = QSLM()
    cgh = CGH(slm=slm)
    wcgh = QCGH(cgh=cgh)
    wcgh.show()
    wcgh.xc = -10
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
