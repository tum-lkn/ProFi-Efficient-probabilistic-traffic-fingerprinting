import logging
import logging.handlers
import os


def get_level(name) -> int:
    if name.find('-worker-') > -1:
        return logging.INFO
    else:
        return logging.DEBUG


def get_log_file(name: str, log_dir: str=None) -> str:
    if log_dir is None:
        log_dir = '/opt/project/data/results/logs'
    else:
        assert os.path.exists(log_dir), f"Directory {log_dir} does not exist!"
    return os.path.join(log_dir, f'{name}.log')


def add_handler(logger: logging.Logger, name: str, log_dir: str=None) -> logging.Logger:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not logger.hasHandlers():
        fh = logging.handlers.RotatingFileHandler(
            get_log_file(name, log_dir),
            mode='w',
            maxBytes=5 * 1024 * 1024,
        )
        fh.setFormatter(formatter)
        fh.setLevel(get_level(name))
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(get_level(name))
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def produce_logger(name: str, log_dir: str=None) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(get_level(name))
        logger = add_handler(logger, name, log_dir)
        return logger


