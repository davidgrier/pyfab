from common.QPropertySheet import QPropertySheet


class QCGHPropertyWidget(QPropertySheet):

    def __init__(self, cgh, cam):
        super(QCGHPropertyWidget, self).__init__(title='CGH Pipeline')
        self.cgh = cgh
        slm = cgh.slm
        register = self.registerProperty
        setter = self.cgh.setter
        self.wxs = register('xs', cgh.rs.x(), 0, slm.width(), setter)
        self.wys = register('ys', cgh.rs.y(), 0, slm.height(), setter)
        self.wqpp = register('qpp', cgh.qpp, 0., 1., setter)
        self.walpha = register('alpha', cgh.alpha, 0.1, 10., setter)
        self.wxc = register('xc', cgh.rc.x(), 0, cam.width(), setter)
        self.wyc = register('yc', cgh.rc.y(), 0, cam.height(), setter)
        self.wzc = register('zc', cgh.rc.z(), -500, 500, setter)
        self.wtheta = register('theta', cgh.theta, -180, 180, setter)
        self.wz0 = register('z0', cgh.z0, 10, 1000, setter)

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
