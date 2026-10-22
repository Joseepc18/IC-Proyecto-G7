"""Une los resultados de cada persona en la tabla unica del contrato (seccion 6)."""
import math
import re
from pathlib import Path

import pandas as pd

from ic_g7 import config

# Metricas del contrato: "media ± std", por ejemplo "0.7674 ± 0.0870".
FORMATO_METRICA = re.compile(r"\d+(\.\d+)? ± \d+(\.\d+)?")
COLUMNAS_METRICA = ["F1_macro_CV (media±std)", "exactitud_CV", "ROC_AUC_CV"]
COLUMNAS_TIEMPO = ["t_entrenamiento", "t_inferencia"]


def _validar_filas(tabla: pd.DataFrame, nombre: str) -> None:
    """Falla con archivo y fila si el contenido no es una fila de resultados valida."""
    if tabla.empty:
        raise ValueError(f"{nombre}: el archivo no tiene filas de resultados")
    for i, fila in tabla.iterrows():
        donde = f"{nombre}, fila {i + 2}"  # +2: encabezado y base 1
        modelo = fila["modelo"]
        if not isinstance(modelo, str) or not modelo.strip():
            raise ValueError(f"{donde}: el nombre del modelo esta vacio")
        for columna in COLUMNAS_METRICA:
            valor = fila[columna]
            if not isinstance(valor, str) or not FORMATO_METRICA.fullmatch(valor):
                raise ValueError(
                    f"{donde}: {columna} debe tener el formato 'media ± std', no {valor!r}"
                )
        for columna in COLUMNAS_TIEMPO:
            try:
                tiempo = float(fila[columna])
            except (TypeError, ValueError):
                tiempo = math.nan
            if not math.isfinite(tiempo) or tiempo < 0:
                raise ValueError(
                    f"{donde}: {columna} debe ser un numero finito no negativo, "
                    f"no {fila[columna]!r}"
                )


def consolidar(directorio: Path = config.DIR_REPORTES) -> pd.DataFrame:
    """Lee directorio/resultados_<nombre>.csv y devuelve una sola tabla.

    Exige las columnas del contrato, filas completas y nombres de modelo
    unicos; los errores nombran el archivo y la fila. No lee
    resultados.csv, que es la salida de esta union y no se edita a mano.
    """
    archivos = sorted(directorio.glob("resultados_*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos resultados_*.csv en {directorio}")
    tablas = []
    for archivo in archivos:
        # round_trip: el lector rapido recorta los ultimos decimales de los tiempos.
        try:
            tabla = pd.read_csv(archivo, float_precision="round_trip")
        except pd.errors.EmptyDataError:
            raise ValueError(f"{archivo.name}: el archivo esta vacio") from None
        if list(tabla.columns) != config.COLUMNAS_RESULTADOS:
            raise ValueError(
                f"{archivo.name}: las columnas no coinciden con las del contrato: "
                f"{config.COLUMNAS_RESULTADOS}"
            )
        _validar_filas(tabla, archivo.name)
        tablas.append(tabla)
    unida = pd.concat(tablas, ignore_index=True)
    repetidos = unida.loc[unida["modelo"].duplicated(), "modelo"].tolist()
    if repetidos:
        raise ValueError(f"Nombres de modelo repetidos entre archivos: {repetidos}")
    return unida
