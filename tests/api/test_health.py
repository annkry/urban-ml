from fastapi.testclient import TestClient

from urban_ml.core.config import settings
from urban_ml.api.main import app

client = TestClient(app)


def test_health_check_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
    }
