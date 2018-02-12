from common.QPropertySheet import QPropertySheet


class QCGHPropertyWidget(QPropertySheet):

    def __init__(self, cgh, cam):
        super(QCGHPropertyWidget, self).__init__(title='CGH Pipeline')
        self.cgh = cgh
        slm = cgh.slm
        register = self.registerProperty
        setter = self.cgh.setProperty
        self.wxs = register('xs', cgh.rs.x(), 0, slm.width(), setter)
        self.wys = register('ys', cgh.rs.y(), 0, slm.height(), setter)
        self.wqpp = register('qpp', cgh.qpp, 0., 1., setter)
        self.walpha = register('alpha', cgh.alpha, 0.1, 10., setter)
        self.wxc = register('xc', cgh.rc.x(), 0, cam.width(), setter)
        self.wyc = register('yc', cgh.rc.y(), 0, cam.height(), setter)
        self.wzc = register('zc', cgh.rc.z(), -500, 500, setter)
        self.wtheta = register('theta', cgh.theta, -180, 180, setter)
        self.wk0 = register('k0', cgh.k0, -10, 10, setter)

    def configuration(self):
        return {'xs': self.wxs.value,
                'ys': self.wys.value,
                'qpp': self.wqpp.value,
                'alpha': self.walpha.value,
                'xc': self.wxc.value,
                'yc': self.wyc.value,
                'zc': self.wzc.value,
                'theta': self.wtheta.value,
                'k0': self.wk0.value}

    def setConfiguration(self, properties):
        for property, value in properties.iteritems():
            try:
                self.__dict__['w'+property].value = value
            except KeyError:
                print('unknown attribute:', property)
