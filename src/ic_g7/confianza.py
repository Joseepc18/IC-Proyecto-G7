"""Calibracion del umbral de baja confianza (contrato, seccion 7).

Trabaja solo con probabilidades fuera de pliegue del entrenamiento: nunca se
le pasan predicciones del conjunto de prueba.
"""
import numpy as np
import pandas as pd

OBJETIVO_EXACTITUD = 0.90
UMBRALES = np.round(np.arange(0.25, 1.0, 0.01), 2)


def tabla_umbrales(probabilidades, y_real, umbrales=UMBRALES) -> pd.DataFrame:
    """Exactitud y cobertura de las predicciones aceptadas (prob. maxima >= umbral).

    Las columnas de ``probabilidades`` deben estar en el orden de las clases
    (0, 1, 2, 3). Si ningun ciclo se acepta, la exactitud queda en NaN.
    """
    probabilidades = np.asarray(probabilidades, dtype=float)
    y_real = np.asarray(y_real)
    maxima = probabilidades.max(axis=1)
    acierto = probabilidades.argmax(axis=1) == y_real
    filas = []
    for umbral in umbrales:
        aceptados = maxima >= umbral
        n = int(aceptados.sum())
        filas.append({
            "umbral": float(umbral),
            "aceptados": n,
            "cobertura": n / len(y_real),
            "exactitud": float(acierto[aceptados].mean()) if n else np.nan,
        })
    return pd.DataFrame(filas)


def elegir_umbral(tabla: pd.DataFrame, objetivo: float = OBJETIVO_EXACTITUD):
    """Devuelve (umbral, alcanzo_objetivo, criterio).

    Si algun umbral logra exactitud >= objetivo, se toma el mas bajo (la mayor
    cobertura posible). Si no, se toma el que maximiza la media armonica entre
    exactitud y cobertura, para no premiar aceptar casi nada ni aceptar todo.
    """
    validos = tabla.dropna(subset=["exactitud"])
    if validos.empty:
        raise ValueError("Ningun umbral acepta ciclos.")
    cumplen = validos[validos["exactitud"] >= objetivo]
    if not cumplen.empty:
        return float(cumplen["umbral"].min()), True, "umbral mas bajo con exactitud >= objetivo"
    armonica = 2 * validos["exactitud"] * validos["cobertura"] / (
        validos["exactitud"] + validos["cobertura"]
    )
    mejor = validos.loc[armonica.idxmax()]
    return float(mejor["umbral"]), False, "mejor media armonica exactitud/cobertura"
