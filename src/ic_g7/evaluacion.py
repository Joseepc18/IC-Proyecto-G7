"""Validacion cruzada por grupos y fila de la tabla unica de resultados (contrato, secciones 4 y 6)."""
from time import perf_counter

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

from ic_g7 import config

PUNTAJES = {"f1_macro": "f1_macro", "exactitud": "accuracy", "roc_auc": "roc_auc_ovr"}


def crear_cv() -> StratifiedGroupKFold:
    return StratifiedGroupKFold(
        n_splits=config.N_PLIEGUES, shuffle=True, random_state=config.RANDOM_STATE
    )


def _media_std(valores) -> str:
    return f"{valores.mean():.4f} ± {valores.std():.4f}"


def validar_cruzado(modelo, X, y, grupos, nombre, hiperparametros=None):
    """Valida con la CV por grupos del contrato.

    Devuelve (fila, por_pliegue): la fila usa las columnas fijas de la tabla de resultados
    y por_pliegue tiene las metricas de cada pliegue. El modelo debe soportar predict_proba.
    t_inferencia se expresa en segundos por lote de validacion (promedio de pliegues).
    """
    pliegues = list(crear_cv().split(X, y, grupos))
    res = cross_validate(
        modelo, X, y, groups=grupos, cv=pliegues, scoring=PUNTAJES,
        n_jobs=1, return_estimator=True,
    )
    # score_time incluye el cálculo de las tres métricas y no mide solo
    # inferencia. Medimos predict_proba del lote de validación por separado.
    tiempos_inferencia = []
    for estimador, (_, indices_validacion) in zip(res["estimator"], pliegues):
        X_validacion = X.iloc[indices_validacion] if hasattr(X, "iloc") else X[indices_validacion]
        inicio = perf_counter()
        estimador.predict_proba(X_validacion)
        tiempos_inferencia.append(perf_counter() - inicio)
    fila = {
        "modelo": nombre,
        "hiperparámetros": str(hiperparametros if hiperparametros is not None else modelo),
        "F1_macro_CV (media±std)": _media_std(res["test_f1_macro"]),
        "exactitud_CV": _media_std(res["test_exactitud"]),
        "ROC_AUC_CV": _media_std(res["test_roc_auc"]),
        "t_entrenamiento": float(res["fit_time"].mean()),
        "t_inferencia": float(sum(tiempos_inferencia) / len(tiempos_inferencia)),
    }
    por_pliegue = pd.DataFrame(
        {
            "pliegue": range(config.N_PLIEGUES),
            "f1_macro": res["test_f1_macro"],
            "exactitud": res["test_exactitud"],
            "roc_auc": res["test_roc_auc"],
        }
    )
    return fila, por_pliegue
