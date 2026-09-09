"""
Abstract interface that every recognition engine (InsightFace, and
potentially LBPH as a fallback) must implement, so the rest of the
application never depends on a specific engine directly.
"""

from abc import ABC, abstractmethod
import numpy as np


class RecognitionAdapter(ABC):
    """Common interface every recognition engine must implement."""

    @abstractmethod
    def detect_faces(self, frame: np.ndarray):
        """Return a list of detected face objects for a given frame."""
        raise NotImplementedError

    @abstractmethod
    def get_embedding(self, face) -> np.ndarray:
        """Return the embedding vector for a detected face."""
        raise NotImplementedError
