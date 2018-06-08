import logging
try:
    from cudaCGH import cudaCGH as CGH
except ImportError, err:
    logging.warning(
        'Could not load CUDA CGH pipeline.\n' +
        '\tFalling back to CPU pipeline.')
    print(err)
    from CGH import CGH
from QCGHPropertyWidget import QCGHPropertyWidget

__all__ = ['CGH', 'QCGHPropertyWidget']
