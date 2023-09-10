import logging
import sys


def _set_backward(logger: logging.Logger) -> None:
    lv = logging.INFO
    logger.setLevel(lv)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(lv)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def _set_forward(logger: logging.Logger) -> None:
    lv = logging.INFO
    logger.setLevel(lv)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(lv)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def set_logging(logger: logging.Logger) -> None:
    {
        "backward": _set_backward,
        "forward": _set_forward
    }[logger.name](logger)


