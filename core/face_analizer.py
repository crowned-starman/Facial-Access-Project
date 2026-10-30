"""
Arquitectura:
  FaceAnalyzer
  ├── detect(frame)         → lista de objetos Face con .bbox, .embedding
  └── get_embedding(face)   → np.ndarray de shape (512,)
"""

import numpy as np
import insightface
from insightface.app import FaceAnalysis

import config
from utils.logger import logger


class FaceAnalyzer:

    def __init__(self):
        logger.info("Inicializando InsightFace (%s) — puede tardar en la primera ejecución...",
                    config.INSIGHTFACE_MODEL_NAME)
        try:
            self._app = FaceAnalysis(name=config.INSIGHTFACE_MODEL_NAME)
            # ctx_id: -1 = CPU | 0, 1... = GPU CUDA
            # det_size: resolución de entrada al detector (mayor = más lento pero detecta caras pequeñas)
            self._app.prepare(
                ctx_id=config.INSIGHTFACE_CTX_ID,
                det_size=config.INSIGHTFACE_DET_SIZE,
            )
            logger.info("InsightFace listo.")
        except Exception as exc:
            logger.exception("Error al inicializar InsightFace: %s", exc)
            raise

    def detect(self, frame: np.ndarray) -> list:

        if frame is None or frame.size == 0:
            logger.warning("Frame vacío recibido en detect().")
            return []

        try:
            faces = self._app.get(frame)
            return faces if faces else []
        except Exception as exc:
            logger.error("Error durante la detección: %s", exc)
            return []

    @staticmethod
    def get_embedding(face) -> np.ndarray:

        if face.embedding is None:
            raise ValueError("El objeto Face no contiene embedding. Verifica que el modelo ArcFace esté cargado.")

        emb = face.embedding.copy()
        # Normalización L2: convierte el vector a norma unitaria para que
        # la similitud coseno sea equivalente al producto punto.
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb
