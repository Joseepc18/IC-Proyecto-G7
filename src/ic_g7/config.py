"""Constantes compartidas. Semilla, rutas y columnas se definen solo aqui."""
from pathlib import Path

RANDOM_STATE = 42
# Valor inicial; se recalibra con predicciones fuera de pliegue (contrato, seccion 7).
UMBRAL_CONFIANZA = 0.87

RAIZ = Path(__file__).resolve().parents[2]
DIR_CRUDO = RAIZ / "data" / "raw" / "hydraulic"
DIR_PROCESADO = RAIZ / "data" / "processed"
RUTA_FEATURES = DIR_PROCESADO / "features.csv"
RUTA_SPLIT = DIR_PROCESADO / "split.csv"
DIR_MODELOS = RAIZ / "models"
DIR_REPORTES = RAIZ / "reports"
# Tabla unica (contrato, seccion 6): cada persona escribe resultados_<nombre>.csv
# y scripts/consolidar_resultados.py genera RUTA_RESULTADOS uniendolos.
RUTA_RESULTADOS = DIR_REPORTES / "resultados.csv"
COLUMNAS_RESULTADOS = [
    "modelo", "hiperparámetros", "F1_macro_CV (media±std)", "exactitud_CV",
    "ROC_AUC_CV", "t_entrenamiento", "t_inferencia",
]

URL_DATOS = (
    "https://archive.ics.uci.edu/static/public/447/"
    "condition+monitoring+of+hydraulic+systems.zip"
)
SHA256_ZIP = "24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2"

SENSORES = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE",
]
ESTADISTICOS = ("mean", "std", "min", "max")
# 17 sensores x 4 estadisticos = 68 variables, en este orden.
COLUMNAS_ENTRADA = [f"{s}_{e}" for s in SENSORES for e in ESTADISTICOS]

COLUMNAS_PERFIL = ["cooler", "valve", "pump_leak", "accumulator", "stable"]
COLUMNAS_CONDICION = COLUMNAS_PERFIL[:4]

ARCHIVOS_ESPERADOS = (
    [f"{s}.txt" for s in SENSORES]
    + ["profile.txt", "description.txt", "documentation.txt"]
)

# Objetivo: presion del acumulador hidraulico (bar), 4 clases ordenadas de menor a mayor presion.
OBJETIVO = "accumulator"
CLASE_POR_OBJETIVO = {90: 0, 100: 1, 115: 2, 130: 3}
NOMBRE_CLASE = {0: "falla_cercana", 1: "muy_reducida", 2: "algo_reducida", 3: "optima"}

N_PLIEGUES = 5
PLIEGUE_PRUEBA = 3

assert len(COLUMNAS_ENTRADA) == 68
