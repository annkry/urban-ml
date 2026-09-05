from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from urban_ml.core.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def get_session() -> Iterator[Session]:
    """Yield a session for the caller to manage transactions with explicitly."""

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency wrapping get_session() for request-scoped sessions."""

    with get_session() as session:
        yield session
