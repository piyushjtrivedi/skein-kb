import logging


class LoggingConfigurator:
    DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    @classmethod
    def configure(cls, level: int = logging.INFO) -> None:
        logging.basicConfig(level=level, format=cls.DEFAULT_FORMAT)
