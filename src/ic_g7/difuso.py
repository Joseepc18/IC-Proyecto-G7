"""Explicacion difusa sencilla de un ciclo (contrato, seccion 7).

Es una EXPLICACION, no un clasificador: describe si tres presiones del ciclo
estan bajas, medias o altas y que nivel de presion del acumulador suele
acompanar ese patron en el entrenamiento. No sustituye a la prediccion de la
MLP. Los limites de "baja/media/alta" salen solo de percentiles del
entrenamiento (nunca de la prueba).

Variables (propuestas en notebooks/01_eda.ipynb, justificadas en
notebooks/05_explicacion_difusa.ipynb): PS5_max, PS3_mean y PS1_mean.
"""
import numpy as np

VARIABLES = ("PS5_max", "PS3_mean", "PS1_mean")
ETIQUETAS = ("baja", "media", "alta")
GRADO_MINIMO = 0.5  # una regla se considera activa desde este grado
CLASE_OPTIMA = 3  # 130 bar


def ajustar_referencia(X) -> dict:
    """Percentiles 25, 50 y 75 de cada variable. Usar SOLO con entrenamiento."""
    return {
        variable: tuple(float(p) for p in np.percentile(X[variable], [25, 50, 75]))
        for variable in VARIABLES
    }


def referencia_desde_entrenamiento() -> dict:
    """Atajo para la app: lee el entrenamiento y devuelve la referencia."""
    from ic_g7.datos import cargar_entrenamiento

    X, _, _ = cargar_entrenamiento()
    return ajustar_referencia(X)


def pertenencia(valor: float, percentiles) -> dict:
    """Grados de pertenencia a baja, media y alta (suman 1).

    baja vale 1 hasta p25 y baja linealmente a 0 en p50; alta vale 0 hasta p50
    y sube a 1 en p75; media es el resto (triangulo con cima en p50).
    """
    p25, p50, p75 = percentiles
    baja = float(np.clip((p50 - valor) / max(p50 - p25, 1e-12), 0.0, 1.0))
    alta = float(np.clip((valor - p50) / max(p75 - p50, 1e-12), 0.0, 1.0))
    return {"baja": baja, "media": 1.0 - baja - alta, "alta": alta}


def grados(fila, referencia: dict) -> dict:
    """Pertenencia de cada variable de la fila (serie o dict con las 68 columnas)."""
    return {v: pertenencia(float(fila[v]), referencia[v]) for v in VARIABLES}


def activacion_reglas(fila, referencia: dict) -> dict:
    """Grado de cada regla (minimo para 'y', maximo para 'o').

    - optima: PS5_max baja Y PS3_mean baja Y PS1_mean baja.
    - menor: PS5_max alta O PS3_mean alta O PS1_mean alta.
    """
    g = grados(fila, referencia)
    return {
        "optima": min(g[v]["baja"] for v in VARIABLES),
        "menor": max(g[v]["alta"] for v in VARIABLES),
    }


def _lista(nombres) -> str:
    nombres = list(nombres)
    if len(nombres) <= 1:
        return "".join(nombres)
    return ", ".join(nombres[:-1]) + " y " + nombres[-1]


def explicar(fila, referencia: dict, clase_predicha=None) -> str:
    """Frase corta en espanol sobre lo que sugieren las tres presiones.

    Si se pasa ``clase_predicha`` (0 a 3) y la explicacion apunta al lado
    contrario, se avisa; la prediccion del modelo no se modifica.
    """
    g = grados(fila, referencia)
    reglas = activacion_reglas(fila, referencia)
    if reglas["optima"] >= GRADO_MINIMO:
        frase = (
            f"{_lista(VARIABLES)} están bajas, lo que apunta a una presión óptima "
            "(130 bar); en el entrenamiento, este patrón fue mucho más frecuente en ese nivel."
        )
        discrepa = clase_predicha is not None and clase_predicha != CLASE_OPTIMA
    elif reglas["menor"] >= GRADO_MINIMO:
        altas = [v for v in VARIABLES if g[v]["alta"] >= GRADO_MINIMO]
        verbo = "está alta" if len(altas) == 1 else "están altas"
        frase = (
            f"{_lista(altas)} {verbo}, lo que apunta a una presión del acumulador "
            "por debajo de 130 bar."
        )
        discrepa = clase_predicha is not None and clase_predicha == CLASE_OPTIMA
    else:
        frase = (
            f"{_lista(VARIABLES)} muestran valores intermedios o mezclados, "
            "así que no indican un nivel de presión claro."
        )
        discrepa = False
    if discrepa:
        frase += " Esta pista no coincide con la predicción del modelo: revisa el ciclo."
    return frase
