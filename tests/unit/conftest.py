from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from urban_ml.storage.models import Base


@pytest.fixture
def session() -> Iterator[Session]:
    """An in-memory SQLite-backed session with the app's schema applied."""

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    db_session = session_factory()
    try:
        yield db_session
    finally:
        db_session.close()
        engine.dispose()
