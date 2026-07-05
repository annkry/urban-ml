from urban_ml.core.config import settings


def test_default_environment():
    assert settings.app_env == "development"
