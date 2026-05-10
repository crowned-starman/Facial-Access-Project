"""
utils/logger.py
===============
Logger centralizado con salida a consola y archivo .log rotativo.
Importa 'logger' desde cualquier módulo del proyecto.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from config import LOGS_DIR

os.makedirs(LOGS_DIR, exist_ok=True)

def get_logger(name: str = "facial_access") -> logging.Logger:
    """
    Crea (o reutiliza) un logger con handlers de consola y archivo.

    Args:
        name: Nombre del logger (aparece en cada línea del log).

    Returns:
        Instancia de logging.Logger configurada.
    """
    log = logging.getLogger(name)

    # Evita duplicar handlers si el logger ya fue inicializado
    if log.handlers:
        return log

    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Consola
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    # Archivo rotativo (máx 2 MB × 3 backups)
    fh = RotatingFileHandler(
        filename=os.path.join(LOGS_DIR, "access.log"),
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log


# Instancia global lista para importar
logger = get_logger()
