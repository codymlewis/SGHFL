import logging
from tqdm import tqdm

logger = logging.getLogger("SmaHFL")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('| %(name)s %(levelname)s @ %(asctime)s in %(filename)s:%(lineno)d | %(message)s'))
ch.setStream(tqdm)
ch.terminator = ""
logger.addHandler(ch)
