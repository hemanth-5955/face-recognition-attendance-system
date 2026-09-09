"""
Concrete RecognitionAdapter implementation backed by InsightFace's
pretrained ArcFace model (buffalo_l). This is the only file in the
project allowed to import insightface directly.
"""

import numpy as np
from insightface.app import FaceAnalysis
from recognition.base import RecognitionAdapter


class InsightFaceAdapter(RecognitionAdapter):
    def __init__(self, model_name: str = "buffalo_l", det_size=(640, 640)):
        self.app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=det_size)

    def detect_faces(self, frame: np.ndarray):
        """Returns InsightFace's own Face objects (already include embeddings)."""
        return self.app.get(frame)

    def get_embedding(self, face) -> np.ndarray:
        """InsightFace computes detection + embedding together, so this
        just extracts it from the face object returned by detect_faces()."""
        return face.embedding
