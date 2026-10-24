"""Carga del modelo final y predicción por ciclo para Streamlit."""
import numpy as np
import pandas as pd

from ic_g7 import config
from ic_g7.datos import cargar_entrenamiento
from ic_g7.difuso import explicar, referencia_desde_entrenamiento
from ic_g7.prototipo import detectar_fuera_de_rango
from ic_g7.modelo_final import cargar_modelo_guardado


def preparar_aplicacion():
    """Carga el artefacto, el ejemplo y la referencia difusa de entrenamiento."""
    modelo, etiquetas, metadata = cargar_modelo_guardado()
    entrenamiento, _, _ = cargar_entrenamiento()
    # Conservamos los límites registrados al preparar el modelo, sin recalcularlos.
    limites = pd.DataFrame.from_dict(
        metadata["rangos_entrenamiento"], orient="index",
    ).loc[config.COLUMNAS_ENTRADA, ["minimo", "maximo"]]
    referencia = referencia_desde_entrenamiento()
    return modelo, etiquetas, entrenamiento, limites, referencia


def predecir_ciclos(modelo, etiquetas, X: pd.DataFrame, limites, referencia) -> pd.DataFrame:
    """Una categoría por fila, con las probabilidades del pipeline ya entrenado."""
    entradas = X.loc[:, config.COLUMNAS_ENTRADA]
    probabilidades = modelo.predict_proba(entradas)
    clases = np.asarray(modelo.classes_)
    if (
        probabilidades.shape != (len(entradas), len(clases))
        or not np.isfinite(probabilidades).all()
        or (probabilidades < 0).any()
        or (probabilidades > 1).any()
        or not np.allclose(probabilidades.sum(axis=1), 1)
    ):
        raise ValueError("El modelo devolvió probabilidades no válidas.")

    # classes_ indica a qué etiqueta corresponde cada columna de predict_proba.
    predichas = clases[probabilidades.argmax(axis=1)]
    salida = pd.DataFrame({
        "Registro": np.arange(1, len(entradas) + 1),
        "Acumulador (bar)": [etiquetas[str(int(c))]["bar"] for c in predichas],
        "Estado": [
            etiquetas[str(int(c))]["estado"].replace("_", " ") for c in predichas
        ],
        "Baja confianza": probabilidades.max(axis=1) < config.UMBRAL_CONFIANZA,
        "Fuera de rango": detectar_fuera_de_rango(entradas, limites).any(axis=1).to_numpy(),
    })
    for indice, clase in enumerate(clases):
        bar = etiquetas[str(int(clase))]["bar"]
        salida[f"{bar} bar (%)"] = probabilidades[:, indice] * 100
    # La regla describe sensores; recibe la clase de la MLP para avisar si discrepa.
    salida["Lectura de sensores"] = [
        explicar(fila, referencia, clase_predicha=int(clase))
        for (_, fila), clase in zip(entradas.iterrows(), predichas)
    ]
    return salida
