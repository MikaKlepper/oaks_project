# pipeline/logger.py

import logging
from pathlib import Path

from tqdm import tqdm


class TqdmLoggingHandler(logging.StreamHandler):
    """
    Logging handler that uses tqdm.write so logs don't break progress bars.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:  # pragma: no cover
            pass


def setup_logger(exp_root, level=logging.INFO):
    """
    Configure root logger with:
      - Console handler using tqdm-friendly output
      - File handler in <exp_root>/logs/pipeline.log
    """
    exp_root = Path(exp_root)
    log_dir = exp_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "pipeline.log"

    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers (useful when re-running inside notebooks)
    if logger.handlers:
        logger.handlers.clear()

    fmt = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Console
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    # File
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.info(f"[Logger] Logging to {log_file}")
