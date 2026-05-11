<div align="center">

# 🎓 Facial Access MVP

**Sistema de reconocimiento facial para control de acceso escolar**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?style=flat-square&logo=opencv)
![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace-orange?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)
![Status](https://img.shields.io/badge/fase-1%20MVP%20local-brightgreen?style=flat-square)

*Fase 1 de un sistema escalable de acceso biométrico. Funciona 100% offline, en CPU, con cualquier webcam.*

</div>

---

## ¿Qué hace este sistema?

Detecta rostros en tiempo real usando tu webcam. Si el rostro pertenece a un usuario registrado, muestra **ACCESS GRANTED**. Si no, **ACCESS DENIED**. Todo corre localmente, sin APIs externas ni conexión a internet.

```
Webcam → RetinaFace (detección) → ArcFace (embedding 512-d) → Similitud coseno → Resultado
```

---

## Demo rápida

```bash
# 1. Registra tu rostro
python register.py
# → Ingresa tu nombre → mira la cámara → listo en segundos

# 2. Inicia el reconocimiento
python recognize.py
# → ACCESS GRANTED / ACCESS DENIED en tiempo real
```

---

## Estructura del proyecto

```
facial_access_mvp/
│
├── config.py                  # ← Todos los parámetros en un solo lugar
├── register.py                # ← Script de registro de usuarios
├── recognize.py               # ← Reconocimiento en tiempo real
│
├── core/                      # Lógica de negocio
│   ├── camera.py              #   Abstracción de webcam (context manager)
│   ├── face_analyzer.py       #   Detección + embeddings (InsightFace/ArcFace)
│   └── embedding_store.py     #   Persistencia y matching por similitud coseno
│
├── utils/                     # Herramientas transversales
│   ├── drawing.py             #   Overlays visuales sobre frames de OpenCV
│   ├── fps_counter.py         #   Contador FPS con ventana deslizante
│   └── logger.py              #   Logger centralizado (consola + archivo rotativo)
│
├── data/
│   ├── embeddings/            #   {username}.npy — array (N × 512) por usuario
│   └── users/                 #   Reservado para fotos y metadatos
│
├── logs/
│   └── access.log             #   Historial de accesos (rotativo, máx 2 MB)
│
└── requirements.txt
```

---

## Requisitos

| Requisito | Versión mínima |
|-----------|---------------|
| Python | 3.9+ |
| Webcam | Cualquier cámara compatible con OpenCV |
| RAM | 4 GB (8 GB recomendado) |
| GPU | Opcional — CUDA acelera la detección |
| Disco | ~600 MB para modelos de InsightFace |

---

## Instalación

```bash
# 1. Clona o descomprime el proyecto
cd facial_access_mvp

# 2. Crea entorno virtual
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Instala dependencias
pip install -r requirements.txt
```

> **Primera ejecución:** InsightFace descargará automáticamente los modelos `buffalo_l` (~500 MB) en `~/.insightface/`. Solo ocurre una vez.

### GPU (opcional)

Si tienes una GPU NVIDIA con CUDA, reemplaza en `requirements.txt`:

```
# Cambia esto:
onnxruntime>=1.16.0

# Por esto:
onnxruntime-gpu>=1.16.0
```

Y en `config.py` ajusta:

```python
INSIGHTFACE_CTX_ID = 0    # 0 = primera GPU CUDA
```

---

## Uso

### Registrar un usuario

```bash
python register.py
```

1. El sistema solicita el nombre del usuario por consola
2. Abre la webcam — mira directamente a la cámara
3. Captura **5 muestras** automáticamente (configurable)
4. Valida que haya **exactamente 1 rostro** visible
5. Guarda el embedding en `data/embeddings/{username}.npy`

> Si el usuario ya existe, el sistema pregunta si deseas sobreescribirlo.

### Reconocimiento en tiempo real

```bash
python recognize.py
```

- Carga todos los usuarios registrados al iniciar
- Muestra bounding box y resultado por cada rostro detectado
- El banner superior indica ACCESS GRANTED / ACCESS DENIED con nombre y score

### Controles de teclado

| Tecla | Acción |
|-------|--------|
| `q` | Salir / cancelar registro |
| `r` | Listar usuarios registrados en consola (solo `recognize.py`) |

---

## Cómo funciona

### 1. Detección — RetinaFace

InsightFace usa **RetinaFace** para localizar rostros en cada frame. Retorna coordenadas `[x1, y1, x2, y2]` y un score de confianza. Los rostros con baja confianza se descartan automáticamente.

### 2. Embedding — ArcFace

**ArcFace** (ResNet-100 con *angular margin loss*) convierte cada rostro en un vector de **512 floats**. Este vector es:
- Único por identidad biométrica
- Robusto a cambios de iluminación y ángulo moderados
- Normalizado a norma unitaria (L2) para simplificar la comparación

### 3. Comparación — Similitud coseno

Con vectores normalizados L2, la similitud coseno se reduce a un producto punto:

```
similitud = embedding_query · embedding_registrado
```

| Valor | Interpretación |
|-------|---------------|
| `~1.0` | Misma persona |
| `~0.55` | Umbral de match (default) |
| `~0.0` | Personas distintas |

### 4. Threshold tuning

Ajusta `SIMILARITY_THRESHOLD` en `config.py` según tu entorno:

| Valor | Cuándo usarlo |
|-------|--------------|
| `0.45` | Iluminación variable, cámara de baja calidad |
| `0.55` | **Default MVP** — buen balance precisión/recall |
| `0.65` | Producción con buena iluminación y cámara HD |

---

## Configuración (`config.py`)

```python
# Cámara
CAMERA_INDEX = 0               # 0 = webcam por defecto

# Modelo (buffalo_l = preciso | buffalo_s = rápido)
INSIGHTFACE_MODEL_NAME = "buffalo_l"
INSIGHTFACE_CTX_ID     = -1    # -1 = CPU | 0 = GPU CUDA

# Registro
REGISTER_SAMPLES      = 5      # Muestras por usuario
REGISTER_DELAY_FRAMES = 10     # Frames de espera entre capturas

# Reconocimiento
SIMILARITY_THRESHOLD  = 0.55   # Umbral de similitud coseno
```

---

## Roadmap

| Fase | Estado | Descripción |
|------|--------|-------------|
| **Fase 1** | ✅ Completa | MVP local — registro y reconocimiento por webcam |
| **Fase 2** | 🔜 Pendiente | Backend FastAPI — REST API para registro y control de acceso |
| **Fase 3** | 🔜 Pendiente | Dashboard web (React) — historial de accesos en tiempo real |
| **Fase 4** | 🔜 Pendiente | Integración torniquete — GPIO / relay controller |
| **Fase 5** | 🔜 Pendiente | Edge AI — ONNX optimizado para Jetson Nano / Raspberry Pi |

### Puntos de extensión ya preparados en el código

| Módulo | En producción, reemplazar por... |
|--------|----------------------------------|
| `core/camera.py` | Stream RTSP / cámara IP |
| `core/embedding_store.py` | PostgreSQL + pgvector / Qdrant |
| `config.py` | Variables de entorno con Pydantic Settings |
| `utils/logger.py` | ELK Stack / CloudWatch |

---

## Troubleshooting

**La cámara no abre**
```bash
# Prueba con diferentes índices en config.py
CAMERA_INDEX = 1   # o 2, 3...
```

**InsightFace no descarga los modelos**
```bash
# Descarga manual forzada:
python -c "from insightface.app import FaceAnalysis; FaceAnalysis('buffalo_l').prepare(ctx_id=-1)"
```

**Falsos positivos — reconoce a personas incorrectas**
```python
# Sube el threshold en config.py
SIMILARITY_THRESHOLD = 0.65
```

**Falsos negativos — no reconoce al usuario registrado**
```python
# Baja el threshold o registra con más muestras
SIMILARITY_THRESHOLD = 0.45
REGISTER_SAMPLES = 10
```

**FPS muy bajo en CPU**
```python
# Cambia al modelo rápido en config.py
INSIGHTFACE_MODEL_NAME = "buffalo_s"
```

---

## Stack tecnológico

| Tecnología | Rol |
|------------|-----|
| **Python 3.9+** | Lenguaje base |
| **OpenCV** | Captura de video y rendering |
| **InsightFace** | Pipeline de detección y reconocimiento facial |
| **ArcFace (ResNet-100)** | Generación de embeddings 512-d |
| **RetinaFace** | Detección de rostros en frames |
| **NumPy** | Álgebra lineal, persistencia `.npy` |
| **ONNX Runtime** | Inferencia optimizada (CPU/GPU) |

---

## Licencia

MIT — libre para uso educativo y comercial con atribución.

---

<div align="center">
Construido como base escalable para sistemas biométricos de control de acceso.
</div>
