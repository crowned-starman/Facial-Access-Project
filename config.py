"""
config.py
=========
Configuración centralizada del sistema de reconocimiento facial.
Modifica aquí los parámetros sin tocar la lógica del negocio.
"""

import os

# ─── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
USERS_DIR      = os.path.join(DATA_DIR, "users")
LOGS_DIR       = os.path.join(BASE_DIR, "logs")

# ─── Cámara ───────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0          # 0 = cámara por defecto; cambia a 1, 2... para cámaras externas
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ─── InsightFace / ArcFace ───────────────────────────────────────────────────
INSIGHTFACE_MODEL_NAME = "buffalo_l"   # buffalo_l = mayor precisión | buffalo_s = más rápido
INSIGHTFACE_CTX_ID     = -1            # -1 = CPU | 0 = GPU (CUDA device 0)
INSIGHTFACE_DET_SIZE   = (640, 640)    # Tamaño de entrada para el detector

# ─── Registro ─────────────────────────────────────────────────────────────────
REGISTER_SAMPLES       = 5             # Número de capturas por usuario
REGISTER_DELAY_FRAMES  = 10            # Frames de espera entre capturas

# ─── Reconocimiento ───────────────────────────────────────────────────────────
# Similitud coseno: 1.0 = idéntico | 0.0 = totalmente diferente
# Ajusta THRESHOLD según tu entorno:
#   - 0.5 → más permisivo (tolerante a cambios de luz/ángulo)
#   - 0.6 → equilibrado (recomendado para MVP)
#   - 0.7 → más estricto (producción con buena iluminación)
SIMILARITY_THRESHOLD   = 0.55

# ─── UI / Display ─────────────────────────────────────────────────────────────
FONT_SCALE        = 0.8
FONT_THICKNESS    = 2
COLOR_GRANTED     = (0, 220, 80)     # Verde
COLOR_DENIED      = (0, 50, 220)     # Rojo (BGR)
COLOR_DETECTING   = (220, 180, 0)    # Azul claro
COLOR_INFO        = (255, 255, 255)  # Blanco
BOX_COLOR         = (0, 200, 255)    # Naranja para bounding box
