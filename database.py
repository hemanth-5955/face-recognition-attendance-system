"""
database.py

SQLite persistence layer for the Face Recognition Attendance System.

Schema overview:
    students            - registered people (identity + course info)
    face_embeddings      - one or more stored facial embeddings per student
    classes              - a named course/class (e.g. "Algorithms")
    attendance_sessions   - one specific meeting of a class, on a given date
    attendance            - a student marked present in a specific session

All functions open and close their own connection; this keeps the module
simple and safe to call from a single process desktop app.
"""

import os
import sqlite3
from datetime import date
import numpy as np
from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "frs.db"

_ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY")
if not _ENCRYPTION_KEY:
    raise RuntimeError(
        "ENCRYPTION_KEY not found. Create a .env file with an ENCRYPTION_KEY "
        "(see README for how to generate one) before running this application."
    )
_cipher = Fernet(_ENCRYPTION_KEY.encode())


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they do not already exist."""
    conn = get_connection()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        course TEXT,
        year TEXT,
        course_duration TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        active INTEGER DEFAULT 1
    );

    CREATE TABLE IF NOT EXISTS face_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        embedding BLOB NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES students(id)
    );

    CREATE TABLE IF NOT EXISTS classes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        course TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS attendance_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        class_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        start_time TEXT,
        end_time TEXT,
        FOREIGN KEY (class_id) REFERENCES classes(id)
    );

    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        student_id INTEGER NOT NULL,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        confidence REAL,
        FOREIGN KEY (session_id) REFERENCES attendance_sessions(id),
        FOREIGN KEY (student_id) REFERENCES students(id)
    );
    """)
    conn.commit()
    conn.close()


def _ensure_column(conn, table, column, coldef):
    """Add a column to an existing table if it isn't already there.
    Used by migrate_db() so older databases created before a schema
    change (e.g. course_duration) can be upgraded in place."""
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coldef}")


def migrate_db():
    """Apply any schema upgrades needed on a database created by an
    older version of init_db(). Safe to call every time on startup."""
    conn = get_connection()
    _ensure_column(conn, "students", "course_duration", "TEXT")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------

def add_student(student_id, name, course=None, year=None, course_duration=None):
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO students (student_id, name, course, year, course_duration) "
        "VALUES (?, ?, ?, ?, ?)",
        (student_id, name, course, year, course_duration),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_student_by_student_id(student_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM students WHERE student_id = ?", (student_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_student(internal_id, name, course, year, course_duration):
    conn = get_connection()
    conn.execute(
        "UPDATE students SET name=?, course=?, year=?, course_duration=? WHERE id=?",
        (name, course, year, course_duration, internal_id),
    )
    conn.commit()
    conn.close()


def set_student_active(internal_id, active: bool):
    conn = get_connection()
    conn.execute("UPDATE students SET active=? WHERE id=?", (1 if active else 0, internal_id))
    conn.commit()
    conn.close()


def delete_student(internal_id):
    """Permanently removes a student along with their stored face
    embeddings and attendance history (required to satisfy foreign keys)."""
    conn = get_connection()
    conn.execute("DELETE FROM attendance WHERE student_id = ?", (internal_id,))
    conn.execute("DELETE FROM face_embeddings WHERE student_id = ?", (internal_id,))
    conn.execute("DELETE FROM students WHERE id = ?", (internal_id,))
    conn.commit()
    conn.close()


def list_students():
    """Active students only. Used wherever "who counts right now" matters."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM students WHERE active = 1").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_all_students_including_inactive():
    """Every student regardless of active status. Used for admin views."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM students ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_students():
    conn = get_connection()
    row = conn.execute("SELECT COUNT(*) as c FROM students WHERE active = 1").fetchone()
    conn.close()
    return row["c"]


# ---------------------------------------------------------------------------
# Face embeddings
# ---------------------------------------------------------------------------

def save_embedding(internal_student_id, embedding: np.ndarray):
    """Embeddings are encrypted before being written to disk, so the
    raw database file cannot be read directly without ENCRYPTION_KEY."""
    raw_bytes = embedding.astype(np.float32).tobytes()
    encrypted_blob = _cipher.encrypt(raw_bytes)
    conn = get_connection()
    conn.execute(
        "INSERT INTO face_embeddings (student_id, embedding) VALUES (?, ?)",
        (internal_student_id, encrypted_blob),
    )
    conn.commit()
    conn.close()


def load_all_embeddings():
    """Returns {student_id: embedding} for every ACTIVE student, using
    each student's most recently captured embedding. Inactive students
    are excluded so they are never matched during live attendance.
    Embeddings are decrypted here using ENCRYPTION_KEY from .env."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT s.student_id, fe.embedding
        FROM face_embeddings fe
        JOIN students s ON s.id = fe.student_id
        WHERE s.active = 1
        AND fe.id IN (SELECT MAX(id) FROM face_embeddings GROUP BY student_id)
    """).fetchall()
    conn.close()
    result = {}
    for row in rows:
        decrypted_bytes = _cipher.decrypt(row["embedding"])
        result[row["student_id"]] = np.frombuffer(decrypted_bytes, dtype=np.float32)
    return result


# ---------------------------------------------------------------------------
# Classes & sessions
# ---------------------------------------------------------------------------

def add_class(name, course=None):
    conn = get_connection()
    cur = conn.execute("INSERT INTO classes (name, course) VALUES (?, ?)", (name, course))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_class_by_name(name):
    conn = get_connection()
    row = conn.execute("SELECT * FROM classes WHERE name = ?", (name,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_or_create_session_for_date(class_id, date_str):
    """Returns the existing session for this class on this date, or
    creates one. Used both for today's live attendance and when a
    record is edited to a different date."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM attendance_sessions WHERE class_id=? AND date=?",
        (class_id, date_str),
    ).fetchone()
    if row:
        session_id = row["id"]
    else:
        cur = conn.execute(
            "INSERT INTO attendance_sessions (class_id, date) VALUES (?, ?)",
            (class_id, date_str),
        )
        session_id = cur.lastrowid
        conn.commit()
    conn.close()
    return session_id


def get_or_create_todays_session(class_id):
    return get_or_create_session_for_date(class_id, date.today().isoformat())


# ---------------------------------------------------------------------------
# Attendance
# ---------------------------------------------------------------------------

def already_marked(session_id, internal_student_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT 1 FROM attendance WHERE session_id = ? AND student_id = ?",
        (session_id, internal_student_id),
    ).fetchone()
    conn.close()
    return row is not None


def mark_attendance(session_id, internal_student_id, confidence):
    """Inserts an attendance record unless this student is already
    marked in this session. Returns True if a new record was created,
    False if it was a duplicate and nothing changed."""
    if already_marked(session_id, internal_student_id):
        return False
    conn = get_connection()
    conn.execute(
        "INSERT INTO attendance (session_id, student_id, confidence) VALUES (?, ?, ?)",
        (session_id, internal_student_id, confidence),
    )
    conn.commit()
    conn.close()
    return True


def duplicate_attendance(attendance_id):
    """Explicitly creates a second attendance row identical to an
    existing one, for cases where the admin genuinely wants a duplicate
    (as opposed to mark_attendance(), which prevents accidental ones)."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM attendance WHERE id=?", (attendance_id,)).fetchone()
    if not row:
        conn.close()
        return None
    cur = conn.execute(
        "INSERT INTO attendance (session_id, student_id, confidence) VALUES (?, ?, ?)",
        (row["session_id"], row["student_id"], row["confidence"]),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def update_attendance_full(attendance_id, class_name, date_str, time_str):
    """Moves a record to a (possibly different) class and date/time,
    creating the class or session if they don't already exist."""
    existing_class = get_class_by_name(class_name)
    class_id = existing_class["id"] if existing_class else add_class(class_name)
    session_id = get_or_create_session_for_date(class_id, date_str)
    timestamp = f"{date_str} {time_str}"
    conn = get_connection()
    conn.execute(
        "UPDATE attendance SET session_id=?, timestamp=? WHERE id=?",
        (session_id, timestamp, attendance_id),
    )
    conn.commit()
    conn.close()


def delete_attendance(attendance_id):
    conn = get_connection()
    conn.execute("DELETE FROM attendance WHERE id = ?", (attendance_id,))
    conn.commit()
    conn.close()


def get_session_attendance(session_id):
    conn = get_connection()
    rows = conn.execute("""
        SELECT s.student_id, s.name, a.timestamp, a.confidence
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        WHERE a.session_id = ?
        ORDER BY a.timestamp
    """, (session_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_all_attendance():
    """Every attendance record across all classes/sessions, most
    recent first. Used by the Attendance Records screen."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT a.id, s.date, st.name, st.student_id, c.name as class_name,
               a.timestamp, a.confidence
        FROM attendance a
        JOIN students st ON st.id = a.student_id
        JOIN attendance_sessions s ON s.id = a.session_id
        JOIN classes c ON c.id = s.class_id
        ORDER BY a.timestamp DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_today_attendance():
    today = date.today().isoformat()
    conn = get_connection()
    row = conn.execute("""
        SELECT COUNT(*) as c FROM attendance a
        JOIN attendance_sessions s ON s.id = a.session_id
        WHERE s.date = ?
    """, (today,)).fetchone()
    conn.close()
    return row["c"]


def get_student_by_internal_id(internal_id):
    """Look up a student by their internal database id (not the
    human readable student_id). Used by admin screens that already
    have the internal id from a table selection."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM students WHERE id = ?", (internal_id,)).fetchone()
    conn.close()
    return dict(row) if row else None
