"""
core/camera.py

"""

import cv2
import numpy as np
from typing import Optional, Tuple

import config
from utils.logger import logger


class Camera:

    def __init__(self, index: int = config.CAMERA_INDEX):
        """
        Args:
            index: Índice de la cámara (0 = default, 1 = externa, etc.)
        """
        self._index = index
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Abre la cámara y configura resolución."""
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            raise IOError(f"No se pudo abrir la cámara con índice {self._index}. "
                          "Verifica que esté conectada y no esté en uso.")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        logger.info("Cámara %d abierta (%dx%d)", self._index, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lee un frame de la cámara.

        Returns:
            (True, frame BGR) si la lectura fue exitosa.
            (False, None) si hubo un error.
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ret, frame = self._cap.read()
        return ret, frame if ret else None

    def release(self) -> None:
        """Libera la cámara y destruye ventanas de OpenCV."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Cámara liberada.")
        cv2.destroyAllWindows()

    # Context Manager
    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()
