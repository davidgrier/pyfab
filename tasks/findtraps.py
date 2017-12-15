from maxtask import maxtask


class findtraps(maxtask):

    def __init__(self, nframes=5):
        super(findtraps, self).__init__(self, nframes)

    def dotask(self):
        print('find traps')
