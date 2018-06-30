import logging
from .QCGHPropertyWidget import QCGHPropertyWidget

useGPU = True
if useGPU:
    try:
        from .cudaCGH import cudaCGH as CGH
    except ImportError as err:
        logging.warning(
            'Could not load CUDA CGH pipeline.\n' +
            '\tFalling back to CPU pipeline.')
        from .CGH import CGH
else:
    from .CGH import CGH


__all__ = ['CGH', 'QCGHPropertyWidget']
