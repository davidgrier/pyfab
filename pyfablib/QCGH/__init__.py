import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from .cupyCGH import cupyCGH as CGH
except Exception as ex:
    logger.warning('Could not import GPU pipeline: {}'.format(ex))
from .CGH import CGH

__all__ = ['CGH']
