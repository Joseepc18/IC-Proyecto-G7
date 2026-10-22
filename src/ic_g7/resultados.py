"""Une los resultados de cada persona en la tabla unica del contrato (seccion 6)."""
from pathlib import Path

import pandas as pd

from ic_g7 import config


def consolidar(directorio: Path = config.DIR_REPORTES) -> pd.DataFrame:
    """Lee directorio/resultados_<nombre>.csv y devuelve una sola tabla.

    Exige las columnas del contrato y nombres de modelo unicos. No lee
    resultados.csv, que es la salida de esta union y no se edita a mano.
    """
    archivos = sorted(directorio.glob("resultados_*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos resultados_*.csv en {directorio}")
    tablas = []
    for archivo in archivos:
        # round_trip: el lector rapido recorta los ultimos decimales de los tiempos.
        tabla = pd.read_csv(archivo, float_precision="round_trip")
        if list(tabla.columns) != config.COLUMNAS_RESULTADOS:
            raise ValueError(
                f"{archivo.name}: las columnas no coinciden con las del contrato: "
                f"{config.COLUMNAS_RESULTADOS}"
            )
        tablas.append(tabla)
    unida = pd.concat(tablas, ignore_index=True)
    repetidos = unida.loc[unida["modelo"].duplicated(), "modelo"].tolist()
    if repetidos:
        raise ValueError(f"Nombres de modelo repetidos entre archivos: {repetidos}")
    return unida
