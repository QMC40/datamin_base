import logging
from logging import Logger
from typing import Optional

from datamin.utils.config import Config


class CLogger(Logger):
    def __init__(self, logger: Optional[Logger], name: str):
        super().__init__(name)
        self.logger = logger

    # Has ovverrides as copying the entire method signature requires some non-importable types
    def info(self, msg: object) -> None:  # type: ignore
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def debug(self, msg: object) -> None:  # type: ignore
        if self.logger is not None:
            self.logger.debug(msg)
        else:
            print(msg)


def get_logger(
    name: str,
    cfg: Config,
    log_dir: Optional[str] = None,
) -> CLogger:

    if cfg.logger_level == "STDOUT":
        logger = CLogger(logger=None, name=name)
    else:
        if cfg.logger_level == "DEBUG":
            level = logging.DEBUG
        elif cfg.logger_level == "INFO":
            level = logging.INFO
        else:
            raise ValueError(f"Invalid logger level: {cfg.logger_level}")
        assert log_dir is not None, "log_dir must be specified"

        handler = logging.FileHandler(log_dir, "w")
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        l_logger = logging.getLogger(name)
        l_logger.setLevel(level)

        l_logger.handlers = []
        l_logger.propagate = False

        l_logger.addHandler(handler)
        logger = CLogger(logger=l_logger, name=name)

    return logger


def get_print_logger(name: str) -> CLogger:
    return CLogger(logger=None, name=name)
