"""Lectura y validación del CSV de sensores para la aplicación."""
import csv
import io

import numpy as np
import pandas as pd

from ic_g7 import config

MAX_BYTES_CSV = 5 * 1024 * 1024


def _primera_celda(mascara: pd.DataFrame) -> str:
    fila, columna = np.argwhere(mascara.to_numpy())[0]
    return f"registro {fila + 1}, columna '{mascara.columns[columna]}'"


def validar_csv(contenido: bytes) -> pd.DataFrame:
    """Lee un CSV de sensores o lanza ValueError con una explicación.

    Exige las 68 columnas, sin etiquetas ni identificadores. Devuelve números
    finitos en el orden de config.COLUMNAS_ENTRADA. El rango se revisa aparte.
    """
    if not contenido or not contenido.strip():
        raise ValueError("El archivo está vacío.")
    if len(contenido) > MAX_BYTES_CSV:
        raise ValueError("El CSV supera el límite de 5 MB.")
    try:
        texto = contenido.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Guarda el archivo como CSV UTF-8, separado por comas.") from exc

    # Revisamos la estructura antes de pandas: evita que renombre columnas
    # duplicadas o interprete como índice una fila con demasiados valores.
    try:
        lector = csv.reader(io.StringIO(texto), strict=True)
        cabecera = next(lector)
        if len(cabecera) != len(set(cabecera)):
            raise ValueError("Hay nombres de columnas duplicados.")
        faltantes = [c for c in config.COLUMNAS_ENTRADA if c not in cabecera]
        adicionales = [c for c in cabecera if c not in config.COLUMNAS_ENTRADA]
        problemas = []
        if faltantes:
            problemas.append("Faltan columnas: " + ", ".join(faltantes[:8]))
        if adicionales:
            problemas.append("Columnas no permitidas: " + ", ".join(adicionales[:8]))
        if problemas:
            raise ValueError(
                ". ".join(problemas)
                + ". Usa el ejemplo descargable con las 68 columnas; "
                "no subas features.csv completo."
            )
        registros = []
        for fila in lector:
            if not fila:  # Las líneas completamente vacías no son registros.
                continue
            if len(fila) != len(cabecera):
                raise ValueError(
                    f"La línea {lector.line_num} no tiene 68 valores. "
                    "Usa comas para separar columnas y punto para decimales."
                )
            registros.append(fila)
    except (csv.Error, StopIteration) as exc:
        raise ValueError("El archivo no tiene una estructura CSV válida.") from exc

    if not registros:
        raise ValueError("El CSV tiene encabezados, pero no contiene ciclos.")

    tabla = pd.DataFrame(registros, columns=cabecera)
    tabla = tabla.loc[:, config.COLUMNAS_ENTRADA]
    nulos = tabla.apply(
        lambda columna: columna.str.strip().str.lower().isin(
            {"", "nan", "na", "n/a", "null", "none"}
        )
    )
    if nulos.to_numpy().any():
        raise ValueError("Hay valores vacíos o nulos: " + _primera_celda(nulos) + ".")

    numeros = tabla.apply(pd.to_numeric, errors="coerce")
    if numeros.isna().to_numpy().any():
        raise ValueError(
            "Hay valores no numéricos: " + _primera_celda(numeros.isna())
            + ". Usa punto para los decimales."
        )
    finitos = pd.DataFrame(
        np.isfinite(numeros.to_numpy(dtype=float)),
        index=numeros.index, columns=numeros.columns,
    )
    if not finitos.to_numpy().all():
        raise ValueError("Hay valores infinitos: " + _primera_celda(~finitos) + ".")

    return numeros.astype(float)


def detectar_fuera_de_rango(datos: pd.DataFrame, limites: pd.DataFrame) -> pd.DataFrame:
    """Marca valores fuera del rango observado sin cambiarlos ni rechazarlos."""
    # Tolerancia mínima para el redondeo de decimales al exportar/importar CSV.
    margen = 1e-12 + 1e-10 * limites.abs().max(axis=1)
    return datos.lt(limites["minimo"] - margen) | datos.gt(limites["maximo"] + margen)
