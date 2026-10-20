"""Construye data/processed/features.csv: 4 estadisticos por sensor y ciclo (68 variables).

Columnas de salida: cycle_id, group_id, cooler (%), clase (0/1/2) y las 68 variables.
La extraccion es fila a fila y no usa etiquetas, asi que no hay fuga de informacion.
"""
import numpy as np
import pandas as pd

from ic_g7 import config


def leer_matriz(ruta) -> np.ndarray:
    return pd.read_csv(ruta, sep="\t", header=None, dtype=np.float64).to_numpy()


def main() -> None:
    perfil = pd.read_csv(
        config.DIR_CRUDO / "profile.txt", sep="\t", header=None, names=config.COLUMNAS_PERFIL
    )
    n = len(perfil)

    columnas = {}
    for sensor in config.SENSORES:
        m = leer_matriz(config.DIR_CRUDO / f"{sensor}.txt")
        if m.shape[0] != n:
            raise ValueError(f"{sensor}: {m.shape[0]} filas, se esperaban {n}")
        if np.isnan(m).any():
            raise ValueError(f"{sensor}: tiene valores nulos")
        columnas[f"{sensor}_mean"] = m.mean(axis=1)
        columnas[f"{sensor}_std"] = m.std(axis=1, ddof=0)  # poblacional, como fija el contrato
        columnas[f"{sensor}_min"] = m.min(axis=1)
        columnas[f"{sensor}_max"] = m.max(axis=1)
        print(f"{sensor}: ok")

    combinacion = perfil[config.COLUMNAS_CONDICION].astype(str).agg("-".join, axis=1)
    base = pd.DataFrame(
        {
            "cycle_id": np.arange(n),
            "group_id": combinacion.astype("category").cat.codes.astype(int),
            "cooler": perfil["cooler"],
            "clase": perfil["cooler"].map(config.CLASE_POR_COOLER),
        }
    )
    features = pd.concat(
        [base, pd.DataFrame(columnas)[config.COLUMNAS_ENTRADA]], axis=1
    )

    config.DIR_PROCESADO.mkdir(parents=True, exist_ok=True)
    features.to_csv(config.RUTA_FEATURES, index=False)
    print(f"Guardado {config.RUTA_FEATURES}: {features.shape[0]} filas x {features.shape[1]} columnas")


if __name__ == "__main__":
    main()
