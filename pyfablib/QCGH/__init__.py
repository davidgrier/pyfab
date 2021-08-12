import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.)

try:
    from .cupyCGH import cupyCGH as CGH
except Exception as ex:
    logger.warning(f'Could not import GPU pipeline: {ex}')
from .CGH import CGH

__all__ = ['CGH']
