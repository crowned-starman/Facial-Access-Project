"""
utils/drawing.py
================
Funciones de dibujo sobre frames de OpenCV.
Centraliza todo lo visual para que sea fácil cambiar el look & feel.
"""

import cv2
import numpy as np
import config


def draw_face_box(frame: np.ndarray, bbox: list, color: tuple = None) -> None:
    """
    Dibuja un rectángulo alrededor del rostro detectado.

    Args:
        frame: Frame BGR de OpenCV (modificado in-place).
        bbox:  [x1, y1, x2, y2] en píxeles.
        color: Tupla BGR. Por defecto usa BOX_COLOR de config.
    """
    color = color or config.BOX_COLOR
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def draw_label(
    frame: np.ndarray,
    text: str,
    position: tuple,
    color: tuple = config.COLOR_INFO,
    bg: bool = True,
) -> None:
    """
    Dibuja texto con fondo semitransparente para legibilidad.

    Args:
        frame:    Frame BGR de OpenCV (modificado in-place).
        text:     Cadena a mostrar.
        position: (x, y) esquina inferior-izquierda del texto.
        color:    Color BGR del texto.
        bg:       Si True, dibuja fondo oscuro detrás del texto.
    """
    font       = cv2.FONT_HERSHEY_SIMPLEX
    scale      = config.FONT_SCALE
    thickness  = config.FONT_THICKNESS

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position

    if bg:
        # Rectángulo de fondo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 4, y - th - 6), (x + tw + 4, y + baseline), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_access_result(
    frame: np.ndarray,
    granted: bool,
    username: str = "",
    similarity: float = 0.0,
) -> None:
    """
    Dibuja el resultado de acceso (GRANTED / DENIED) en la parte superior del frame.

    Args:
        frame:      Frame BGR de OpenCV.
        granted:    True → acceso permitido, False → denegado.
        username:   Nombre del usuario reconocido (solo si granted=True).
        similarity: Puntuación de similitud coseno.
    """
    h, w = frame.shape[:2]

    if granted:
        status_text = "ACCESS GRANTED"
        status_color = config.COLOR_GRANTED
        user_text = f"  {username}  |  sim: {similarity:.2f}"
    else:
        status_text = "ACCESS DENIED"
        status_color = config.COLOR_DENIED
        user_text = "Unknown"

    # Barra de estado en la parte superior
    cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
    draw_label(frame, status_text, (12, 36), color=status_color, bg=False)
    draw_label(frame, user_text,  (w // 2, 36), color=config.COLOR_INFO, bg=False)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """
    Muestra el contador de FPS en la esquina inferior derecha.

    Args:
        frame: Frame BGR de OpenCV.
        fps:   Valor calculado de frames por segundo.
    """
    h, w = frame.shape[:2]
    text = f"FPS: {fps:.1f}"
    draw_label(frame, text, (w - 120, h - 14), color=config.COLOR_INFO)


def draw_register_overlay(
    frame: np.ndarray,
    sample_count: int,
    total_samples: int,
    username: str,
) -> None:
    """
    Overlay informativo durante el proceso de registro.

    Args:
        frame:         Frame BGR de OpenCV.
        sample_count:  Muestras capturadas hasta ahora.
        total_samples: Total de muestras requeridas.
        username:      Nombre del usuario en registro.
    """
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
    draw_label(frame, f"REGISTERING: {username}", (12, 34), color=config.COLOR_DETECTING, bg=False)
    draw_label(
        frame,
        f"Sample {sample_count}/{total_samples}  |  Press 'q' to cancel",
        (12, h - 14),
        color=config.COLOR_INFO,
    )
