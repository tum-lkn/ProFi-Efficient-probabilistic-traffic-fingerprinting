import logging


def logger_setup(name: str, level: int) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fh = logging.FileHandler(f"/mnt/TRACES/logs/{name}.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def produce(name: str) -> logging.Logger:
    configured = {
        "Supervisor": lambda: logger_setup(name, logging.DEBUG),
        "Container": lambda: logger_setup(name, logging.DEBUG)
    }
    tmp = name.split('-')[0]
    if tmp in configured:
        return configured[tmp]()
    else:
        return logger_setup(name, logging.DEBUG)
