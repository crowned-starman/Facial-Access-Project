"""
recognize.py
============
FASE 2 — Reconocimiento facial en tiempo real para control de acceso.

Flujo por frame:
  1. Captura frame de webcam.
  2. Detecta rostros con RetinaFace (vía InsightFace).
  3. Para cada rostro, genera embedding ArcFace (512-d, normalizado L2).
  4. Compara contra embeddings registrados usando similitud coseno.
  5. Si similitud ≥ threshold → ACCESS GRANTED (muestra nombre).
     Si no → ACCESS DENIED.

Uso:
  python recognize.py

Controles:
  q → Salir
  r → Mostrar usuarios registrados en consola
"""

import sys
import os
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from core.camera          import Camera
from core.face_analyzer   import FaceAnalyzer
from core.embedding_store import EmbeddingStore
from utils.drawing        import (
    draw_face_box,
    draw_label,
    draw_access_result,
    draw_fps,
)
from utils.fps_counter    import FPSCounter
from utils.logger         import logger


WINDOW_NAME = "Facial Access — Reconocimiento"


def run_recognition() -> None:
    """
    Loop principal de reconocimiento en tiempo real.

    Arquitectura del loop:
      Frame → detect() → [faces] → para cada face:
        get_embedding() → find_match() → draw resultado
    """
    logger.info("Iniciando sistema de reconocimiento...")

    # Inicialización de componentes
    analyzer = FaceAnalyzer()
    store    = EmbeddingStore()
    fps      = FPSCounter()

    if len(store) == 0:
        logger.warning("No hay usuarios registrados. Ejecuta register.py primero.")
        print("\n  ⚠️  No hay usuarios registrados.")
        print("  Ejecuta: python register.py\n")
        return

    logger.info("Usuarios cargados: %s", store.users)
    print(f"\n  Sistema activo. Usuarios registrados: {store.users}")
    print("  Presiona 'q' para salir | 'r' para listar usuarios\n")

    with Camera() as cam:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                logger.error("Error leyendo frame de la cámara.")
                break

            fps.tick()
            faces = analyzer.detect(frame)

            # Sin rostros detectados
            if not faces:
                draw_label(frame, "Buscando rostros...", (20, 80), color=config.COLOR_DETECTING)
                draw_fps(frame, fps.get())
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    print(f"  Usuarios registrados: {store.users}")
                continue

            # Procesa cada rostro detectado 
            # Nota: en control de acceso normalmente se espera 1 sola persona.
            # Este loop soporta múltiples para mayor flexibilidad futura.
            granted_any = False

            for face in faces:
                # 1. Genera el embedding del rostro actual
                embedding = analyzer.get_embedding(face)

                # 2. Busca el mejor match en el store
                #    - Calcula similitud coseno contra todas las muestras de cada usuario
                #    - Retorna (username, score) o (None, score)
                username, similarity = store.find_match(
                    embedding,
                    threshold=config.SIMILARITY_THRESHOLD
                )

                granted = username is not None

                # 3. Dibuja bounding box
                box_color = config.COLOR_GRANTED if granted else config.COLOR_DENIED
                draw_face_box(frame, face.bbox, color=box_color)

                # 4. Etiqueta encima del bounding box
                x1, y1 = int(face.bbox[0]), int(face.bbox[1])
                label   = f"{username} ({similarity:.2f})" if granted else f"Unknown ({similarity:.2f})"
                label_color = config.COLOR_GRANTED if granted else config.COLOR_DENIED
                draw_label(frame, label, (x1, max(y1 - 10, 20)), color=label_color)

                if granted:
                    granted_any = True
                    logger.info("ACCESO PERMITIDO → %s (sim=%.3f)", username, similarity)
                else:
                    logger.info("ACCESO DENEGADO (mejor sim=%.3f)", similarity)

            #  Banner global de resultado
            # Muestra el resultado del primer rostro en el banner superior
            primary_face       = faces[0]
            primary_embedding  = analyzer.get_embedding(primary_face)
            primary_name, primary_sim = store.find_match(primary_embedding, config.SIMILARITY_THRESHOLD)

            draw_access_result(
                frame,
                granted=primary_name is not None,
                username=primary_name or "",
                similarity=primary_sim,
            )

            draw_fps(frame, fps.get())
            cv2.imshow(WINDOW_NAME, frame)

            # Controles de teclado
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Sistema detenido por el usuario.")
                break
            elif key == ord("r"):
                print(f"  Usuarios registrados: {store.users}")


# Entry point
if __name__ == "__main__":
    try:
        run_recognition()
    except KeyboardInterrupt:
        print("\n  Sistema detenido.")
    except Exception as exc:
        logger.exception("Error inesperado en reconocimiento: %s", exc)
        sys.exit(1)
