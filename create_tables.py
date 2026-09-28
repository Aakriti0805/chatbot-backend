from app.database import Base, DATABASE_URL, engine
from app import models  # noqa: F401
from sqlalchemy import text


def create_database_objects() -> None:
    with engine.begin() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS chatbot"))
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(bind=engine)
    print(f"Database tables created successfully for {DATABASE_URL.split('@')[-1]}")


if __name__ == "__main__":
    create_database_objects()