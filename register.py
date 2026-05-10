"""
register.py
===========
FASE 1 — Registro de usuarios en el sistema de control de acceso.

Flujo:
  1. Solicita nombre del usuario por consola.
  2. Abre webcam y detecta rostros en tiempo real.
  3. Valida que haya exactamente 1 rostro visible.
  4. Captura N muestras (configurable en config.py).
  5. Genera embeddings ArcFace por muestra.
  6. Guarda el array de embeddings en data/embeddings/{username}.npy

Uso:
  python register.py

Controles:
  q → Cancelar registro
"""

import sys
import os
import numpy as np
import cv2

# Asegura que el directorio raíz esté en el path (necesario si se ejecuta desde otra carpeta)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from core.camera          import Camera
from core.face_analyzer   import FaceAnalyzer
from core.embedding_store import EmbeddingStore
from utils.drawing        import draw_face_box, draw_label, draw_register_overlay
from utils.fps_counter    import FPSCounter
from utils.logger         import logger


# Constantes 
WINDOW_NAME     = "Facial Access — Registro"
SAMPLES_NEEDED  = config.REGISTER_SAMPLES
FRAME_DELAY     = config.REGISTER_DELAY_FRAMES   # frames de espera entre capturas


def get_username() -> str:
    """Solicita y valida el nombre del usuario por consola."""
    print("\n" + "=" * 50)
    print("  SISTEMA DE REGISTRO FACIAL")
    print("=" * 50)
    while True:
        name = input("Ingresa el nombre del usuario: ").strip()
        if not name:
            print("  El nombre no puede estar vacío.")
            continue
        # Caracteres seguros para nombre de archivo
        safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip()
        if not safe_name:
            print("  Usa solo letras, números, guiones o espacios.")
            continue
        # Reemplaza espacios con guión bajo para el nombre de archivo
        return safe_name.replace(" ", "_")


def register_user(username: str) -> bool:
    """
    Proceso principal de registro.

    Args:
        username: Nombre limpio del usuario (sin caracteres especiales).

    Returns:
        True si el registro se completó correctamente, False si fue cancelado.
    """
    analyzer = FaceAnalyzer()
    store    = EmbeddingStore()
    fps      = FPSCounter()

    # Avisa si el usuario ya existe
    if username in store.users:
        confirm = input(f"\n  Usuario '{username}' ya existe. ¿Sobreescribir? (s/n): ").lower()
        if confirm != "s":
            logger.info("Registro cancelado por el usuario.")
            return False

    collected_embeddings: list = []   # Lista de arrays (512,)
    frame_counter = 0                 # Contador de frames para el delay entre capturas

    logger.info("Iniciando registro para: %s (%d muestras requeridas)", username, SAMPLES_NEEDED)
    print(f"\n  Mira directamente a la cámara. Se capturarán {SAMPLES_NEEDED} muestras.")
    print("  Presiona 'q' para cancelar.\n")

    with Camera() as cam:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                logger.error("No se pudo leer frame de la cámara.")
                break

            fps.tick()
            faces = analyzer.detect(frame)

            # Validaciones
            if len(faces) == 0:
                draw_label(frame, "No se detecta rostro", (20, 80), color=config.COLOR_DENIED)

            elif len(faces) > 1:
                # Solo un rostro durante el registro para evitar ambigüedad
                draw_label(
                    frame,
                    f"Demasiados rostros detectados ({len(faces)}). Regístrate solo.",
                    (20, 80),
                    color=config.COLOR_DENIED,
                )
                for face in faces:
                    draw_face_box(frame, face.bbox, color=config.COLOR_DENIED)

            else:
                # Exactamente 1 rostro
                face = faces[0]
                draw_face_box(frame, face.bbox, color=config.COLOR_GRANTED)
                frame_counter += 1

                if frame_counter >= FRAME_DELAY:
                    # Captura la muestra
                    embedding = analyzer.get_embedding(face)
                    collected_embeddings.append(embedding)
                    frame_counter = 0
                    logger.debug("Muestra %d/%d capturada.", len(collected_embeddings), SAMPLES_NEEDED)
                    print(f"  ✓ Muestra {len(collected_embeddings)}/{SAMPLES_NEEDED} capturada")

                # Barra de progreso visual
                progress = len(collected_embeddings) / SAMPLES_NEEDED
                h, w = frame.shape[:2]
                bar_w = int(w * progress)
                cv2.rectangle(frame, (0, h - 8), (bar_w, h), config.COLOR_GRANTED, -1)

            # Overlay informativo
            draw_register_overlay(frame, len(collected_embeddings), SAMPLES_NEEDED, username)

            # FPS
            from utils.drawing import draw_fps
            draw_fps(frame, fps.get())

            cv2.imshow(WINDOW_NAME, frame)

            # Salida con 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Registro cancelado por el usuario (tecla 'q').")
                return False

            # Fin del registro 
            if len(collected_embeddings) >= SAMPLES_NEEDED:
                break

    if not collected_embeddings:
        logger.error("No se recopilaron muestras. Registro fallido.")
        return False

    # Guarda el array de shape (N, 512) en disco
    embeddings_array = np.stack(collected_embeddings, axis=0)
    store.save_user(username, embeddings_array)

    logger.info("Registro completado: %s — %d muestras guardadas.", username, len(embeddings_array))
    print(f"\n  ✅ Registro exitoso: '{username}' ({len(embeddings_array)} muestras guardadas)")
    return True


#  Entry point
if __name__ == "__main__":
    try:
        username = get_username()
        success  = register_user(username)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n  Registro interrumpido.")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Error inesperado durante el registro: %s", exc)
        sys.exit(1)
