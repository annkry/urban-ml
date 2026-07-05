import logging

from urban_ml.core.logging import get_logger


def test_get_logger_returns_logger() -> None:
    logger = get_logger("urban_ml.test")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "urban_ml.test"
