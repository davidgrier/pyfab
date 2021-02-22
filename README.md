# pyfab

**pyfab** is a python GUI for a holographic optical trapping system.
When integrated with appropriate hardware, it presents a live video
image of the system being controlled together with graphical representations
of traps that are being projected into that system.  Traps can be
interactively added and deleted, grouped and translated.

**jansen** is the video front end for **pyfab** and stands alone as a
light-weight interface for holographic video microscopy.

## Installation

#### Without hardware
1. Start with a fresh python 3.6 installation or virtual environment (strongly recommended).

2. Install dependencies from requirements.txt: 
      `pip install -r requirements.txt`

3. If you have CUDA installed on an nVidia GPU, you may install [cupy](https://github.com/cupy/cupy)>=7.0.0 for CUDA acceleration.

4. If you wish to add Lorenz-Mie machine vision capability, install [pylorenzmie](https://github.com/davidgrier/pylorenzmie) and [CNNLorenzMie](https://github.com/laltman2/CNNLorenzMie).



# Vision

This branch is deprecated, but contains useful vision code. For now, our real-time tracking code QVision will be removed from master, until pylorenzmie is finished updating and until pylorenzmie and CNNLorenzMie can be more cleanly implemented into pyfab. In the furure, code in this branch specific to Vision's UI, tracking organization and data handling can be moved back into master.
