"""Almacenamiento local de embeddings para un MVP de investigación.

Los archivos NumPy NO están cifrados. Requieren protección adicional del host.
"""
import os
import re
import tempfile
from typing import Dict, Optional, Tuple

import numpy as np

import config
from utils.logger import logger


class EmbeddingStore:
    DIMENSION = 512
    MAX_SAMPLES = 100
    USERNAME_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}", re.ASCII)

    def __init__(self):
        os.makedirs(config.EMBEDDINGS_DIR, mode=0o700, exist_ok=True)
        self._store: Dict[str, np.ndarray] = {}
        self._load_all()

    @classmethod
    def _validate_username(cls, username: str) -> None:
        if not isinstance(username, str) or not cls.USERNAME_PATTERN.fullmatch(username):
            raise ValueError("Identificador inválido: usa 1-64 letras ASCII, números, _ o -.")

    @classmethod
    def _validate_embeddings(cls, embeddings: np.ndarray) -> np.ndarray:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if (vectors.ndim != 2 or vectors.shape[1] != cls.DIMENSION
                or not 1 <= vectors.shape[0] <= cls.MAX_SAMPLES):
            raise ValueError("Los embeddings deben tener forma (N, 512), con 1 <= N <= 100.")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("Embeddings con NaN o infinito.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("Embedding de norma cero.")
        return vectors / norms

    def _load_all(self) -> None:
        for fname in os.listdir(config.EMBEDDINGS_DIR):
            if not fname.endswith(".npy"):
                continue
            username = os.path.splitext(fname)[0]
            path = os.path.join(config.EMBEDDINGS_DIR, fname)
            try:
                self._validate_username(username)
                if os.path.islink(path):
                    raise ValueError("No se aceptan enlaces simbólicos.")
                vectors = np.load(path, allow_pickle=False)
                self._store[username] = self._validate_embeddings(vectors)
            except (OSError, ValueError, TypeError) as exc:
                logger.error("Se ignoró archivo biométrico inválido '%s': %s", fname, exc)
        logger.info("Usuarios registrados en memoria: %d", len(self._store))

    def save_user(self, username: str, embeddings: np.ndarray) -> None:
        self._validate_username(username)
        vectors = self._validate_embeddings(embeddings)
        path = os.path.join(config.EMBEDDINGS_DIR, username + ".npy")
        if os.path.islink(path):
            raise ValueError("No se permite sobrescribir un enlace simbólico.")
        fd, temporary_path = tempfile.mkstemp(prefix=".embedding-", suffix=".tmp",
                                               dir=config.EMBEDDINGS_DIR)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.save(stream, vectors, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            if os.name == "posix":
                os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

        self._store[username] = vectors
        logger.info("Embeddings guardados para '%s' (%d muestras)", username, len(vectors))

    def delete_user(self, username: str) -> bool:
        self._validate_username(username)
        if username not in self._store:
            return False
        path = os.path.join(config.EMBEDDINGS_DIR, username + ".npy")
        if os.path.islink(path):
            raise ValueError("No se permite eliminar un enlace simbólico.")
        if os.path.exists(path):
            os.remove(path)
        del self._store[username]
        logger.info("Usuario eliminado: %s", username)
        return True

    def find_match(
        self,
        query_embedding: np.ndarray,
        threshold: float = config.SIMILARITY_THRESHOLD,
    ) -> Tuple[Optional[str], float]:
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.shape != (self.DIMENSION,) or not np.all(np.isfinite(query)):
            raise ValueError("Embedding de consulta inválido.")
        norm = float(np.linalg.norm(query))
        if norm <= 1e-12:
            raise ValueError("Embedding de consulta de norma cero.")
        if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError("El umbral debe estar en (0, 1].")
        query = query / norm
        if not self._store:
            return None, 0.0

        best_name: Optional[str] = None
        best_score = -1.0
        for username, vectors in self._store.items():
            score = float(np.max(vectors @ query))
            if score > best_score:
                best_name, best_score = username, score

        if best_score >= threshold:
            return best_name, best_score
        return None, best_score

    @property
    def users(self) -> list:
        return list(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)
