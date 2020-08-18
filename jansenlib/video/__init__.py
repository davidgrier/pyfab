import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    from .QSpinnaker.QSpinnaker import QSpinnaker as QCamera
except Exception as ex:
    logger.warning('Could not import Spinnaker camera: {}'.format(ex))
    from .QOpenCV.QOpenCV import QOpenCV as QCamera

__all__ = ['QCamera']
