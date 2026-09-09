import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import database as db


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_frs.db")
    monkeypatch.setattr(db, "DB_PATH", db_path)
    db.init_db()
    db.migrate_db()
    return db
