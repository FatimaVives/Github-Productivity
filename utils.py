# utils.py
import logging
from pathlib import Path
from config import LOG_DIR

def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("github_prod")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        fh = logging.FileHandler(Path(LOG_DIR)/"app.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
