from common.QPropertySheet import QPropertySheet


class QFabVideo(QPropertySheet):

    def __init__(self, camera=None):
        super(QFabVideo, self).__init__(title='Video Camera', header=False)
        self.camera = camera
        self.wmirrored = self.registerProperty('mirrored', self.mirrored)
        self.wflipped = self.registerProperty('flipped', self.flipped)
        self.wtransposed = self.registerProperty('transposed', self.transposed)
        self.wgray = self.registerProperty('gray', self.gray)
        self.wmirrored.valueChanged.connect(self.updateMirrored)
        self.wflipped.valueChanged.connect(self.updateFlipped)
        self.wtransposed.valueChanged.connect(self.updateTransposed)
        self.wgray.valueChanged.connect(self.updateGray)

    def updateMirrored(self):
        self.camera.mirrored = self.wmirrored.value

    @property
    def mirrored(self):
        return self.camera.mirrored

    @mirrored.setter
    def mirrored(self, state):
        value = bool(state)
        self.wmirrored.value = value
        self.updateMirrored()

    def updateFlipped(self):
        self.camera.flipped = self.wflipped.value

    @property
    def flipped(self):
        return self.camera.flipped

    @flipped.setter
    def flipped(self, state):
        value = bool(state)
        self.wflipped.value = value
        self.updateFlipped()

    def updateTransposed(self):
        self.camera.transposed = self.wtransposed.value

    @property
    def transposed(self):
        return self.camera.transposed

    @transposed.setter
    def transposed(self, state):
        value = bool(state)
        self.wtransposed.value = value
        self.updateTransposed()

    def updateGray(self):
        self.camera.gray = self.wgray.value

    @property
    def gray(self):
        return self.camera.gray

    @gray.setter
    def gray(self, state):
        value = bool(state)
        self.wgray.value = value
        self.updateGray()
