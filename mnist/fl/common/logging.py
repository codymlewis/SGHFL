"""
Logging handling made by this library.
"""


import logging


__LOGGER = logging.getLogger("flagon")
__LOGGER.setLevel(logging.DEBUG)

__ch = logging.StreamHandler()
__ch.setLevel(logging.DEBUG)
__ch.setFormatter(logging.Formatter('| %(name)s %(levelname)s @ %(asctime)s in %(filename)s:%(lineno)d | %(message)s'))

__LOGGER.addHandler(__ch)
logger = __LOGGER
