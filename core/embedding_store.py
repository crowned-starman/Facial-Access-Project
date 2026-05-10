"""
core/embedding_store.py
=======================
Gestión persistente de embeddings faciales en disco.

Estructura en disco:
  data/embeddings/
  └── {username}.npy   ← array de shape (N, 512) con N muestras del usuario

¿Por qué múltiples muestras por usuario?
  Promediar N embeddings capturados en condiciones ligeramente distintas
  (ángulo, luz) hace el embedding representativo más robusto.
  En producción esto se reemplazaría por una base de datos vectorial (Qdrant, Weaviate...).
"""

import os
import numpy as np
from typing import Dict, Optional, Tuple

import config
from utils.logger import logger


class EmbeddingStore:
    """
    CRUD de embeddings en archivos .npy por usuario.

    Attributes:
        _store: Dict[str, np.ndarray] — mapa username → array (N, 512) en memoria.
    """

    def __init__(self):
        os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)
        self._store: Dict[str, np.ndarray] = {}
        self._load_all()

    # Persistencia

    def _load_all(self) -> None:
        """Carga en memoria todos los embeddings guardados en disco al iniciar."""
        files = [f for f in os.listdir(config.EMBEDDINGS_DIR) if f.endswith(".npy")]
        for fname in files:
            username = os.path.splitext(fname)[0]
            path = os.path.join(config.EMBEDDINGS_DIR, fname)
            try:
                self._store[username] = np.load(path)
                logger.debug("Embeddings cargados: %s (%d muestras)", username, len(self._store[username]))
            except Exception as exc:
                logger.error("Error cargando embeddings de '%s': %s", username, exc)

        logger.info("Usuarios registrados en memoria: %d", len(self._store))

    def save_user(self, username: str, embeddings: np.ndarray) -> None:
        """
        Guarda (o sobreescribe) los embeddings de un usuario en disco.

        Args:
            username:   Nombre único del usuario.
            embeddings: Array de shape (N, 512) con N muestras.
        """
        self._store[username] = embeddings
        path = os.path.join(config.EMBEDDINGS_DIR, f"{username}.npy")
        np.save(path, embeddings)
        logger.info("Embeddings guardados → %s (%d muestras)", path, len(embeddings))

    def delete_user(self, username: str) -> bool:
        """
        Elimina un usuario del store y del disco.

        Returns:
            True si se eliminó correctamente, False si no existía.
        """
        if username not in self._store:
            return False
        del self._store[username]
        path = os.path.join(config.EMBEDDINGS_DIR, f"{username}.npy")
        if os.path.exists(path):
            os.remove(path)
            logger.info("Usuario eliminado: %s", username)
        return True


    # Matching

    def find_match(
        self,
        query_embedding: np.ndarray,
        threshold: float = config.SIMILARITY_THRESHOLD,
    ) -> Tuple[Optional[str], float]:
        """
        Compara query_embedding contra todos los registros y retorna el mejor match.

        Algoritmo:
          1. Para cada usuario, calcula la similitud coseno entre query_embedding
             y CADA una de sus N muestras.
          2. Toma el máximo (similitud pico) → más robusto que el promedio.
          3. Retorna el usuario con mayor similitud si supera el threshold.

        Similitud coseno:
          cos(θ) = (A · B) / (||A|| × ||B||)
          Como ambos vectores están normalizados L2, reduce a: A · B (producto punto)
          Rango: [-1, 1] donde 1 = idéntico, 0 = ortogonal, -1 = opuesto.

        Args:
            query_embedding: Vector 512-d normalizado del rostro a identificar.
            threshold:       Mínima similitud para considerar match.

        Returns:
            Tupla (username, similarity) si hay match.
            Tupla (None, best_similarity) si nadie supera el threshold.
        """
        if not self._store:
            logger.warning("No hay usuarios registrados en el store.")
            return None, 0.0

        best_name  : Optional[str] = None
        best_score : float         = -1.0

        for username, stored_embeddings in self._store.items():
            # stored_embeddings shape: (N, 512)
            # Producto punto entre query (512,) y cada muestra (512,) → array (N,)
            similarities = stored_embeddings @ query_embedding
            peak_sim     = float(np.max(similarities))

            if peak_sim > best_score:
                best_score = peak_sim
                best_name  = username

        if best_score >= threshold:
            logger.info("Match: %s (sim=%.3f, threshold=%.2f)", best_name, best_score, threshold)
            return best_name, best_score

        logger.debug("Sin match. Mejor score: %.3f < threshold %.2f", best_score, threshold)
        return None, best_score

    # Utilidades


    @property
    def users(self) -> list:
        """Retorna lista de usernames registrados."""
        return list(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)
