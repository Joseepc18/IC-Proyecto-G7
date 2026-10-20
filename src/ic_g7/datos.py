"""Carga de variables y particion. Todos leen la misma particion desde split.csv."""
import pandas as pd

from ic_g7 import config


def cargar_features() -> pd.DataFrame:
    """Columnas: cycle_id, group_id, accumulator, clase y las 68 variables de entrada."""
    if not config.RUTA_FEATURES.exists():
        raise FileNotFoundError(
            "Falta features.csv. Ejecuta: python scripts/descargar_datos.py "
            "y luego python scripts/construir_features.py"
        )
    return pd.read_csv(config.RUTA_FEATURES)


def _cargar_particion(nombre: str):
    if not config.RUTA_SPLIT.exists():
        raise FileNotFoundError("Falta data/processed/split.csv (viene en el repo).")
    features = cargar_features()
    split = pd.read_csv(config.RUTA_SPLIT)
    if len(split) != len(features) or not (split["cycle_id"] == features["cycle_id"]).all():
        raise ValueError("features.csv y split.csv no tienen los mismos ciclos.")
    if not (split["group_id"] == features["group_id"]).all():
        raise ValueError(
            "Los group_id de features.csv no coinciden con split.csv: "
            "regenera features.csv con scripts/construir_features.py."
        )
    parte = features[(split["partition"] == nombre).to_numpy()]
    return (
        parte[config.COLUMNAS_ENTRADA].reset_index(drop=True),
        parte["clase"].reset_index(drop=True),
        parte["group_id"].reset_index(drop=True),
    )


def cargar_entrenamiento():
    """Devuelve X, y, grupos del conjunto de entrenamiento (1.816 ciclos)."""
    return _cargar_particion("train")


def cargar_prueba():
    """Devuelve X, y, grupos de prueba. USAR UNA SOLA VEZ, con el modelo final ya elegido."""
    return _cargar_particion("test")
