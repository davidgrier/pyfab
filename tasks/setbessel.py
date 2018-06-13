# -*- coding: utf-8 -*-
# MENU: Set Bessel

from task import task
import numpy as np


class bessel(task):
    def __init__(self, Modetot=1, **kwargs):
        super(bessel, self).__init__(**kwargs)
        self.Modetot = Modetot

    def dotask(self):
        cgh = self.parent.cgh
        shift=.08
        width=768
        height=1024 #find out SLM size (also height/width are reversed)
        Nx=np.arange(1, width) #idea: do -width/4, width/4 (after making width 2x)
        Ny=np.arange(1, height) #same for height - creates large enough image to be able to drag around maybe
        xcoord=Nx - width/2
        ycoord=-(Ny - height/2)
        [Xc,Yc]=np.meshgrid(xcoord,ycoord)

        Modetot=1

        shift0=0
        phi = np.remainder(np.angle(self.Modetot)-shift*(np.sqrt(Xc**2+Yc**2))-shift0*(Xc),2*(np.pi))
        cgh.setPhi(((255./(2.*np.pi))*phi).astype(np.uint8))


class setbessel(task):
    """Set Bessel"""

    def __init__(self, **kwargs):
        super(setbessel, self).__init__(**kwargs)
        self.kwargs = kwargs

    def dotask(self):
        self.register('cleartraps')
        self.register(bessel(**self.kwargs))
