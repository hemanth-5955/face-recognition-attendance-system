import numpy as np


def test_add_and_get_student(test_db):
    internal_id = test_db.add_student("T001", "Test One", "CS", "2", "3")
    assert internal_id is not None
    student = test_db.get_student_by_student_id("T001")
    assert student is not None
    assert student["name"] == "Test One"


def test_list_students_only_active(test_db):
    test_db.add_student("T001", "Active Student")
    inactive_id = test_db.add_student("T002", "Inactive Student")
    test_db.set_student_active(inactive_id, False)
    names = [s["name"] for s in test_db.list_students()]
    assert "Active Student" in names
    assert "Inactive Student" not in names


def test_update_student(test_db):
    internal_id = test_db.add_student("T001", "Old Name", "CS", "1", "3")
    test_db.update_student(internal_id, "New Name", "Maths", "2", "4")
    student = test_db.get_student_by_student_id("T001")
    assert student["name"] == "New Name"
    assert student["course"] == "Maths"


def test_delete_student(test_db):
    internal_id = test_db.add_student("T001", "To Delete")
    test_db.delete_student(internal_id)
    assert test_db.get_student_by_student_id("T001") is None


def test_save_and_load_embedding(test_db):
    internal_id = test_db.add_student("T001", "Embed Test")
    fake_embedding = np.random.rand(512).astype(np.float32)
    test_db.save_embedding(internal_id, fake_embedding)
    loaded = test_db.load_all_embeddings()
    assert "T001" in loaded
    assert loaded["T001"].shape == (512,)


def test_inactive_student_excluded_from_embeddings(test_db):
    internal_id = test_db.add_student("T001", "Will Be Inactive")
    fake_embedding = np.random.rand(512).astype(np.float32)
    test_db.save_embedding(internal_id, fake_embedding)
    test_db.set_student_active(internal_id, False)
    loaded = test_db.load_all_embeddings()
    assert "T001" not in loaded


def test_attendance_marked_and_duplicate_prevented(test_db):
    internal_id = test_db.add_student("T001", "Attend Test")
    class_id = test_db.add_class("Test Class")
    session_id = test_db.get_or_create_todays_session(class_id)

    first = test_db.mark_attendance(session_id, internal_id, 0.9)
    second = test_db.mark_attendance(session_id, internal_id, 0.9)

    assert first is True
    assert second is False
    assert len(test_db.get_session_attendance(session_id)) == 1


def test_duplicate_attendance_function_creates_new_row(test_db):
    internal_id = test_db.add_student("T001", "Dup Test")
    class_id = test_db.add_class("Test Class")
    session_id = test_db.get_or_create_todays_session(class_id)
    test_db.mark_attendance(session_id, internal_id, 0.9)

    before = test_db.list_all_attendance()
    test_db.duplicate_attendance(before[0]["id"])
    after = test_db.list_all_attendance()

    assert len(after) == len(before) + 1


def test_delete_attendance(test_db):
    internal_id = test_db.add_student("T001", "Delete Attend Test")
    class_id = test_db.add_class("Test Class")
    session_id = test_db.get_or_create_todays_session(class_id)
    test_db.mark_attendance(session_id, internal_id, 0.9)

    record_id = test_db.list_all_attendance()[0]["id"]
    test_db.delete_attendance(record_id)

    assert len(test_db.list_all_attendance()) == 0
