import time
from collections import deque


class FPSCounter:

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
        
        if not self._times:
            return 0.0
        avg_delta = sum(self._times) / len(self._times)
        return 1.0 / avg_delta if avg_delta > 0 else 0.0
