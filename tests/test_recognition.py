import os
import numpy as np
import pytest
import cv2

from recognition.insightface_adapter import InsightFaceAdapter
from recognition.matcher import match_embedding, cosine_similarity

ROOT = os.path.join(os.path.dirname(__file__), "..")
TEST_FACE = os.path.join(ROOT, "test_face.jpg")
STRANGER_FACE = os.path.join(ROOT, "stranger.jpg")


@pytest.fixture(scope="module")
def adapter():
    return InsightFaceAdapter()


@pytest.mark.skipif(not os.path.exists(TEST_FACE), reason="test_face.jpg not present")
def test_detects_one_face(adapter):
    img = cv2.imread(TEST_FACE)
    faces = adapter.detect_faces(img)
    assert len(faces) == 1


@pytest.mark.skipif(not os.path.exists(TEST_FACE), reason="test_face.jpg not present")
def test_embedding_is_512_dimensions(adapter):
    img = cv2.imread(TEST_FACE)
    faces = adapter.detect_faces(img)
    embedding = adapter.get_embedding(faces[0])
    assert embedding.shape == (512,)


@pytest.mark.skipif(not os.path.exists(TEST_FACE), reason="test_face.jpg not present")
def test_same_face_matches_itself(adapter):
    img = cv2.imread(TEST_FACE)
    faces = adapter.detect_faces(img)
    embedding = adapter.get_embedding(faces[0])

    known = {"student_001": embedding}
    student_id, score = match_embedding(embedding, known)

    assert student_id == "student_001"
    assert score > 0.99


@pytest.mark.skipif(
    not (os.path.exists(TEST_FACE) and os.path.exists(STRANGER_FACE)),
    reason="test_face.jpg or stranger.jpg not present",
)
def test_different_face_does_not_match(adapter):
    known_img = cv2.imread(TEST_FACE)
    stranger_img = cv2.imread(STRANGER_FACE)

    known_faces = adapter.detect_faces(known_img)
    stranger_faces = adapter.detect_faces(stranger_img)

    known_embedding = adapter.get_embedding(known_faces[0])
    stranger_embedding = adapter.get_embedding(stranger_faces[0])

    known = {"student_001": known_embedding}
    student_id, score = match_embedding(stranger_embedding, known)

    assert student_id is None


def test_cosine_similarity_identical_vectors():
    vec = np.random.rand(512).astype(np.float32)
    score = cosine_similarity(vec, vec)
    assert abs(score - 1.0) < 0.001


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    score = cosine_similarity(a, b)
    assert abs(score - 0.0) < 0.001
