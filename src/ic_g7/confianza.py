"""Calibracion del umbral de baja confianza (contrato, seccion 7).

Trabaja solo con probabilidades fuera de pliegue del entrenamiento: nunca se
le pasan predicciones del conjunto de prueba.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_predict

OBJETIVO_EXACTITUD = 0.90
UMBRALES = np.round(np.arange(0.25, 1.0, 0.01), 2)


def validar_entradas(probabilidades, y_real):
    """Devuelve (probabilidades, y_real) como arreglos o lanza ValueError claro."""
    probabilidades = np.asarray(probabilidades, dtype=float)
    y_real = np.asarray(y_real)
    if probabilidades.ndim != 2 or probabilidades.shape[0] == 0:
        raise ValueError("Las probabilidades deben ser una matriz no vacia (ciclos x clases).")
    if y_real.ndim != 1 or len(y_real) != len(probabilidades):
        raise ValueError(
            f"Debe haber una etiqueta por ciclo: {len(probabilidades)} filas de "
            f"probabilidades y {y_real.size} etiquetas."
        )
    if not np.isfinite(probabilidades).all():
        raise ValueError("Las probabilidades contienen valores no finitos.")
    if (probabilidades < 0).any() or (probabilidades > 1).any():
        raise ValueError("Las probabilidades deben estar entre 0 y 1.")
    if not np.allclose(probabilidades.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Las probabilidades de cada ciclo deben sumar 1.")
    n_clases = probabilidades.shape[1]
    if (not np.issubdtype(y_real.dtype, np.integer)
            or y_real.min() < 0 or y_real.max() >= n_clases):
        raise ValueError(f"Las etiquetas deben ser enteros entre 0 y {n_clases - 1}.")
    return probabilidades, y_real


def probabilidades_fuera_de_pliegue(modelo, X, y, grupos, pliegues) -> np.ndarray:
    """Probabilidades fuera de pliegue; se detiene si el modelo no converge.

    Las probabilidades de una MLP a medio entrenar contaminarian el umbral,
    asi que un aviso de convergencia es un error y no se calibra nada.
    """
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        probabilidades = cross_val_predict(
            modelo, X, y, groups=grupos, cv=pliegues, method="predict_proba", n_jobs=1,
        )
    convergencia = [a for a in avisos if issubclass(a.category, ConvergenceWarning)]
    if convergencia:
        raise RuntimeError(
            f"Hubo {len(convergencia)} avisos de convergencia; no se calibra el umbral."
        )
    for aviso in avisos:  # no ocultar otros avisos
        warnings.warn(str(aviso.message), aviso.category, stacklevel=2)
    return probabilidades


def tabla_umbrales(probabilidades, y_real, umbrales=UMBRALES) -> pd.DataFrame:
    """Exactitud y cobertura de las predicciones aceptadas (prob. maxima >= umbral).

    Las columnas de ``probabilidades`` deben estar en el orden de las clases
    (0, 1, 2, 3). Si ningun ciclo se acepta, la exactitud queda en NaN. Las
    entradas se validan antes de calcular (ver ``validar_entradas``).
    """
    probabilidades, y_real = validar_entradas(probabilidades, y_real)
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
