"""
Logging handling made by this library.
"""


import logging
from tqdm import tqdm


__LOGGER = logging.getLogger("SmaHFL")
__LOGGER.setLevel(logging.DEBUG)

__ch = logging.StreamHandler()
__ch.setLevel(logging.DEBUG)
__ch.setFormatter(logging.Formatter('| %(name)s %(levelname)s @ %(asctime)s in %(filename)s:%(lineno)d | %(message)s'))
__ch.setStream(tqdm)
__ch.terminator = ""

__LOGGER.addHandler(__ch)
logger = __LOGGER
