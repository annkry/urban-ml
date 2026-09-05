from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from urban_ml.storage.models import Base


@pytest.fixture
def session() -> Iterator[Session]:
    """An in-memory SQLite-backed session with the app's schema applied.

    StaticPool + check_same_thread=False: FastAPI's TestClient runs sync
    endpoints in a worker thread, and an in-memory sqlite db is otherwise
    tied to the single connection that created it — a second thread would
    silently get a fresh, empty database rather than an error.
    """

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    db_session = session_factory()
    try:
        yield db_session
    finally:
        db_session.close()
        engine.dispose()
