"""Crea data/processed/split.csv, la particion unica train/test del proyecto.

Esta particion ya viene en el repo: no hace falta ejecutarlo. Solo se usa --forzar si el equipo
acuerda cambiarla, y entonces hay que subir el nuevo archivo para todos.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ic_g7 import config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forzar", action="store_true", help="sobrescribe split.csv")
    args = parser.parse_args()

    if config.RUTA_SPLIT.exists() and not args.forzar:
        raise SystemExit("split.csv ya existe: es la referencia comun. Usa --forzar solo si el equipo lo acuerda.")

    features = pd.read_csv(config.RUTA_FEATURES, usecols=["cycle_id", "group_id", "clase"])
    y = features["clase"].to_numpy()
    grupos = features["group_id"].to_numpy()

    cv = StratifiedGroupKFold(
        n_splits=config.N_PLIEGUES, shuffle=True, random_state=config.RANDOM_STATE
    )
    pliegues = list(cv.split(np.zeros(len(y)), y, grupos))
    idx_train, idx_test = pliegues[config.PLIEGUE_PRUEBA]

    if set(grupos[idx_train]) & set(grupos[idx_test]):
        raise RuntimeError("Fuga: hay grupos en entrenamiento y prueba")

    particion = np.where(np.isin(np.arange(len(y)), idx_test), "test", "train")
    split = pd.DataFrame(
        {"cycle_id": features["cycle_id"], "group_id": features["group_id"], "partition": particion}
    )
    split.to_csv(config.RUTA_SPLIT, index=False)

    for nombre, idx in (("train", idx_train), ("test", idx_test)):
        conteo = np.bincount(y[idx], minlength=4).tolist()
        print(f"{nombre}: {len(idx)} ciclos, clases [0, 1, 2, 3] = {conteo}")
    print(f"Guardado {config.RUTA_SPLIT}")


if __name__ == "__main__":
    main()
