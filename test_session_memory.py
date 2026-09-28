from app.database import SessionLocal
from app.services.query_services import _get_or_create_conversation


def test_session_id_reuses_same_conversation():
    db = SessionLocal()
    try:
        first = _get_or_create_conversation(db, None, "browser-session-abc")
        second = _get_or_create_conversation(db, None, "browser-session-abc")
        assert first.id == second.id, "same session_id should reuse the same conversation"
    finally:
        db.close()


if __name__ == "__main__":
    test_session_id_reuses_same_conversation()
    print("session-memory test passed")
