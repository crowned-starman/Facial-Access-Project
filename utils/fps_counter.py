"""
utils/fps_counter.py
====================
Contador de FPS con ventana deslizante para suavizar el valor mostrado.
"""

import time
from collections import deque


class FPSCounter:
    """
    Calcula FPS promedio usando una ventana de las últimas N mediciones.

    Uso:
        fps = FPSCounter(window=30)
        while True:
            fps.tick()
            print(fps.get())
    """

    def __init__(self, window: int = 30):
        """
        Args:
            window: Número de frames para calcular el promedio deslizante.
        """
        self._times: deque = deque(maxlen=window)
        self._last: float  = time.perf_counter()

    def tick(self) -> None:
        """Registra el tiempo del frame actual."""
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now

    def get(self) -> float:
        """
        Retorna el FPS promedio actual.

        Returns:
            FPS como float. Retorna 0.0 si no hay datos aún.
        """
        if not self._times:
            return 0.0
        avg_delta = sum(self._times) / len(self._times)
        return 1.0 / avg_delta if avg_delta > 0 else 0.0
