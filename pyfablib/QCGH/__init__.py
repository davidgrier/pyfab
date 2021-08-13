import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

try:
    from .cupyCGH import cupyCGH as CGH
    logger.info('Using cupyCGH pipeline')
except Exception as ex:
    logger.warning('Could not import cupyCGH pipeline: {}'.format(ex))
    from .CGH import CGH


__all__ = ['CGH']
