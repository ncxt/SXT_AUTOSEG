"""
sxtcnn
"""

import logging

logging.basicConfig(
    format="%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)

from .sxtcnn import SXTCNN

