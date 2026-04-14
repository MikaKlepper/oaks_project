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


def _emit_config_line(message: str) -> None:
    logger = logging.getLogger()
    if logger.handlers:
        logging.info(message)
    else:
        print(message)


def log_config_resolution(cfg, split_dir) -> None:
    target_mode = cfg.data.target_mode
    target_column = cfg.data.get("target_column")
    target_finding = cfg.data.get("target_finding")
    positive_value = cfg.data.get("target_positive_value")

    _emit_config_line(
        f"[Config] Dataset={cfg.datasets.name} | Split={cfg.datasets.split} | "
        f"TargetTask={cfg.data.target_task}"
    )
    _emit_config_line(
        f"[Config] Calibration -> enabled={cfg.calibration.enabled}, "
        f"num_samples={cfg.calibration.get('num_samples')}, "
        f"base_dataset={cfg.calibration.get('base_dataset')}, "
        f"init_from_base={cfg.calibration.get('init_from_base')}"
    )
    _emit_config_line(
        f"[Config] Target definition -> mode={target_mode}, "
        f"column={target_column}, finding={target_finding}, positive_value={positive_value}"
    )
    _emit_config_line(f"[Config] Split directory -> {split_dir}")
    _emit_config_line(f"[Config] Active subset CSV -> {cfg.datasets.get('subset_csv')}")
    _emit_config_line("[Config] Feature backend -> feature_bank")
    _emit_config_line(f"[Config] Feature bank root -> {cfg.features.bank_root}")
    _emit_config_line(f"[Config] Feature registry -> {cfg.features.registry_path}")
    _emit_config_line(f"[Config] Local bank root -> {cfg.features.get('local_bank_root')}")
    _emit_config_line(f"[Config] Experiment tag -> {cfg.experiment.tag}")
