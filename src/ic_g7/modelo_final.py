"""Reentrenamiento del modelo elegido y evaluación reservada del proyecto.

Preparar el artefacto usa solo entrenamiento. La prueba se carga únicamente en
evaluar_prueba_una_vez, con confirmación explícita de la evaluación única.
"""
import ast
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from ic_g7 import config
from ic_g7.datos import cargar_entrenamiento
from ic_g7.pipeline import crear_pipeline

NOMBRE_SELECCIONADO = "MLP+GA (semilla 42)"
PARAMETROS_SELECCIONADOS = {
    "hidden_layer_sizes": (64,),
    "alpha": 0.0035127047262708444,
    "learning_rate_init": 0.0026698666742744587,
    "activation": "tanh",
    "max_iter": 4000,
    "random_state": config.RANDOM_STATE,
}
ARCHIVOS_MODELO = ("pipeline_final.joblib", "labels.json", "metadata_final.json")
BAR_POR_CLASE = {clase: bar for bar, clase in config.CLASE_POR_OBJETIVO.items()}
CLASES = sorted(BAR_POR_CLASE)


def _ahora():
    return datetime.now(timezone.utc).isoformat()


def _guardar_json(ruta, contenido):
    Path(ruta).write_text(
        json.dumps(contenido, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _hash_archivo(ruta):
    return sha256(Path(ruta).read_bytes()).hexdigest()


def _firma_entrenamiento(X, y, grupos):
    """Incluye datos, etiquetas, grupos y orden; no consulta la partición de prueba."""
    h = sha256()
    for tabla in (X, y, grupos):
        h.update(pd.util.hash_pandas_object(tabla, index=True).to_numpy().tobytes())
    h.update(json.dumps(list(X.columns)).encode("utf-8"))
    return h.hexdigest()


def leer_seleccion():
    """Contrasta la configuración comunicada con la fila original ya integrada."""
    fuente = config.DIR_REPORTES / "resultados_jose.csv"
    tabla = pd.read_csv(fuente)
    filas = tabla.loc[tabla["modelo"] == NOMBRE_SELECCIONADO]
    if len(filas) != 1:
        raise ValueError("Debe existir una única fila de la configuración elegida.")
    fila = filas.iloc[0]
    parametros = ast.literal_eval(fila["hiperparámetros"])
    if parametros != PARAMETROS_SELECCIONADOS:
        raise ValueError("La configuración registrada no coincide con la indicada por José.")
    media, desviacion = map(float, fila["F1_macro_CV (media±std)"].split(" ± "))
    return {
        "modelo": NOMBRE_SELECCIONADO,
        "origen": "Selección comunicada por José y solicitada por el equipo en esta tarea.",
        "fuente": "reports/resultados_jose.csv",
        "fuente_sha256": _hash_archivo(fuente),
        "hiperparametros": {**parametros, "hidden_layer_sizes": list(parametros["hidden_layer_sizes"])},
        "f1_macro_cv_media": media,
        "f1_macro_cv_std": desviacion,
        "nota": "Media y desviación entre pliegues, redondeadas como en el CSV. "
                "Elegir esta configuración no demuestra que GA supere a búsqueda aleatoria.",
    }


def crear_modelo_elegido():
    """Pipeline compartido: escalado, filtro aprendido y MLP de la selección."""
    return crear_pipeline(MLPClassifier(**PARAMETROS_SELECCIONADOS))


def _validar_entradas(X, y, grupos):
    if list(X.columns) != config.COLUMNAS_ENTRADA:
        raise ValueError("Se requieren las 68 entradas acordadas, en su orden original.")
    if not (len(X) == len(y) == len(grupos)) or not len(X):
        raise ValueError("Entradas, etiquetas y grupos deben tener igual longitud y no estar vacíos.")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Hay entradas no numéricas o no finitas.")
    if sorted(y.unique().tolist()) != CLASES or grupos.isna().any():
        raise ValueError("Se requieren las cuatro clases acordadas y grupos sin nulos.")


def preparar_modelo(directorio=None):
    """Ajusta con todo train y guarda un pipeline, etiquetas y trazabilidad.

    Rechaza sobrescrituras. El modelo queda reentrenado y pendiente de la
    evaluación única en prueba, no certificado para diagnosticar una máquina.
    """
    destino = Path(directorio) if directorio is not None else config.DIR_MODELOS
    if any((destino / nombre).exists() for nombre in ARCHIVOS_MODELO):
        raise FileExistsError("Ya hay artefactos finales; no se sobrescriben ni se reentrena a ciegas.")
    seleccion = leer_seleccion()
    X, y, grupos = cargar_entrenamiento()
    _validar_entradas(X, y, grupos)
    hash_split = _hash_archivo(config.RUTA_SPLIT)
    firma_train = _firma_entrenamiento(X, y, grupos)
    modelo = crear_modelo_elegido()
    inicio = perf_counter()
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        modelo.fit(X, y)
    if any(issubclass(a.category, ConvergenceWarning) for a in avisos):
        raise RuntimeError("La MLP no convergió: no se guardan artefactos finales.")
    for aviso in avisos:
        warnings.warn(str(aviso.message), aviso.category, stacklevel=2)
    segundos = perf_counter() - inicio
    if _hash_archivo(config.RUTA_SPLIT) != hash_split:
        raise RuntimeError("La partición cambió durante el entrenamiento.")
    np.testing.assert_array_equal(modelo.classes_, CLASES)
    etiquetas = {
        str(clase): {"bar": BAR_POR_CLASE[clase], "estado": config.NOMBRE_CLASE[clase]}
        for clase in CLASES
    }
    destino.mkdir(parents=True, exist_ok=True)
    # Primero se comprueba la serialización; metadata se publica al final.
    with TemporaryDirectory(prefix=".preparacion-", dir=destino) as temporal:
        temporal = Path(temporal)
        ruta_modelo = temporal / ARCHIVOS_MODELO[0]
        joblib.dump(modelo, ruta_modelo, compress=3)
        recuperado = joblib.load(ruta_modelo)
        with threadpool_limits(limits=1):
            np.testing.assert_allclose(
                recuperado.predict_proba(X), modelo.predict_proba(X), rtol=0, atol=0,
            )
        _guardar_json(temporal / "labels.json", etiquetas)
        metadata = {
            "schema_version": 1, "fecha_utc": _ahora(),
            "estado": "reentrenado; evaluación de prueba pendiente al generar estos artefactos",
            "seleccion": seleccion, "split_sha256": hash_split,
            "entrenamiento_sha256": firma_train,
            "columnas_entrada": config.COLUMNAS_ENTRADA,
            "clases_ordenadas": CLASES,
            "ciclos_entrenamiento": len(X), "grupos_entrenamiento": int(grupos.nunique()),
            "distribucion_clases": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
            "variables_conservadas": modelo[:-1].get_feature_names_out().tolist(),
            "rangos_entrenamiento": {
                c: {"minimo": float(X[c].min()), "maximo": float(X[c].max())}
                for c in config.COLUMNAS_ENTRADA
            },
            "segundos_reentrenamiento": segundos,
            "iteraciones_mlp": int(modelo.named_steps["modelo"].n_iter_),
            "avisos_convergencia": 0, "prueba_usada_en_reentrenamiento": False,
            "pipeline_sha256": _hash_archivo(ruta_modelo),
            "labels_sha256": _hash_archivo(temporal / "labels.json"),
            "python": platform.python_version(),
            "versiones": {p: version(p) for p in ("scikit-learn", "numpy", "pandas", "joblib")},
            "limite_hilos_blas": 1,
            "pendiente": "Evaluación única con el equipo e integración en Streamlit. "
                         "El umbral está disponible en config.py y las tendencias en difuso.py; "
                         "son complementos del pipeline, no parámetros de su entrenamiento.",
        }
        _guardar_json(temporal / "metadata_final.json", metadata)
        for nombre in ARCHIVOS_MODELO:
            os.replace(temporal / nombre, destino / nombre)
    return metadata


def cargar_modelo_guardado(directorio=None):
    """Carga solo artefactos locales de confianza y comprueba su integridad."""
    origen = Path(directorio) if directorio is not None else config.DIR_MODELOS
    metadata = json.loads((origen / "metadata_final.json").read_text(encoding="utf-8"))
    if metadata["versiones"] != {
        p: version(p) for p in ("scikit-learn", "numpy", "pandas", "joblib")
    }:
        raise ValueError("Las versiones actuales no coinciden con las del modelo guardado.")
    for nombre, campo in (
        ("pipeline_final.joblib", "pipeline_sha256"), ("labels.json", "labels_sha256"),
    ):
        if _hash_archivo(origen / nombre) != metadata[campo]:
            raise ValueError(f"El archivo {nombre} no coincide con su huella registrada.")
    etiquetas = json.loads((origen / "labels.json").read_text(encoding="utf-8"))
    esperadas = {
        str(c): {"bar": BAR_POR_CLASE[c], "estado": config.NOMBRE_CLASE[c]} for c in CLASES
    }
    if etiquetas != esperadas or metadata["columnas_entrada"] != config.COLUMNAS_ENTRADA:
        raise ValueError("Las etiquetas o columnas del modelo no respetan el contrato.")
    # joblib se usa únicamente con el artefacto local generado por este proyecto.
    modelo = joblib.load(origen / "pipeline_final.joblib")
    np.testing.assert_array_equal(modelo.classes_, CLASES)
    return modelo, etiquetas, metadata


def evaluar_prueba_una_vez(confirmar_evaluacion=False, directorio_modelos=None,
                           directorio_reportes=None):
    """Evalúa el artefacto congelado, sin fit ni selección de hiperparámetros.

    El registro se crea de forma exclusiva ANTES de cargar prueba. Un fallo tras
    ese momento también impide repetirla automáticamente: se revisa con el equipo.
    """
    if not confirmar_evaluacion:
        raise PermissionError("La prueba se reserva a la evaluación final acordada. "
                              "Se requiere confirmación explícita de la evaluación única.")
    destino = (Path(directorio_reportes) if directorio_reportes is not None
               else config.DIR_REPORTES / "evaluacion_final")
    registro = destino / "registro.json"
    if registro.exists():
        raise FileExistsError("La evaluación ya fue iniciada o completada; no se repite.")
    modelo, etiquetas, metadata = cargar_modelo_guardado(directorio_modelos)
    X_train, y_train, grupos_train = cargar_entrenamiento()
    if (_hash_archivo(config.RUTA_SPLIT) != metadata["split_sha256"]
            or _firma_entrenamiento(X_train, y_train, grupos_train)
            != metadata["entrenamiento_sha256"]):
        raise ValueError("Los datos o la partición cambiaron desde el reentrenamiento.")
    destino.mkdir(parents=True, exist_ok=True)
    estado = {
        "fecha_inicio_utc": _ahora(), "estado": "iniciada",
        "modelo_sha256": metadata["pipeline_sha256"],
        "split_sha256": metadata["split_sha256"],
        "confirmacion": "Quien ejecuta confirma la evaluación única acordada y la selección previa del modelo.",
    }
    with registro.open("x", encoding="utf-8") as archivo:
        json.dump(estado, archivo, ensure_ascii=False, indent=2)
    try:
        from ic_g7.datos import cargar_prueba
        X, y, grupos = cargar_prueba()
        _validar_entradas(X, y, grupos)
        if set(grupos) & set(grupos_train):
            raise ValueError("Hay grupos compartidos entre entrenamiento y prueba.")
        inicio = perf_counter()
        with threadpool_limits(limits=1):
            probabilidades = modelo.predict_proba(X)
        segundos = perf_counter() - inicio
        if (probabilidades.shape != (len(y), len(CLASES))
                or not np.isfinite(probabilidades).all()
                or (probabilidades < 0).any() or (probabilidades > 1).any()
                or not np.allclose(probabilidades.sum(axis=1), 1)):
            raise ValueError("Las probabilidades del modelo no son válidas.")
        prediccion = np.asarray(modelo.classes_)[probabilidades.argmax(axis=1)]
        precision, recall, f1, soporte = precision_recall_fscore_support(
            y, prediccion, labels=CLASES, zero_division=0,
        )
        matriz = confusion_matrix(y, prediccion, labels=CLASES)
        f1_macro = float(f1_score(y, prediccion, average="macro"))
        seleccion = metadata["seleccion"]
        limite_b = seleccion["f1_macro_cv_media"] - 2 * seleccion["f1_macro_cv_std"]
        informe = {
            "fecha_utc": _ahora(), "modelo": seleccion["modelo"],
            "pipeline_sha256": metadata["pipeline_sha256"],
            "ciclos_prueba": len(X), "grupos_prueba": int(grupos.nunique()),
            "f1_macro_prueba": f1_macro,
            "exactitud_prueba": float(accuracy_score(y, prediccion)),
            "roc_auc_ovr_macro_prueba": float(roc_auc_score(y, probabilidades, multi_class="ovr")),
            "segundos_inferencia_lote": segundos,
            "segundos_reentrenamiento": metadata["segundos_reentrenamiento"],
            "orden_matriz_bar": [BAR_POR_CLASE[c] for c in CLASES],
            "matriz_confusion": matriz.tolist(),
            "criterio_b": {
                "limite_cv_menos_2_std": limite_b,
                "cumple": f1_macro >= limite_b,
                "nota": "Referencia de CV redondeada del CSV; no se ajusta el modelo "
                        "ni se repite prueba si el criterio no se cumple.",
            },
            "confianza": {
                "umbral": float(config.UMBRAL_CONFIANZA),
                "origen": "Probabilidades OOF de entrenamiento; no se recalibra con prueba.",
                "tendencias": "Las reglas difusas describen sensores y no explican cómo decide la MLP.",
            },
        }
        metricas = pd.DataFrame({
            "clase_bar": [BAR_POR_CLASE[c] for c in CLASES],
            "precision": precision, "recall": recall, "f1": f1, "soporte": soporte,
        })
        predicciones = pd.DataFrame({
            "indice_prueba": np.arange(len(y)), "group_id": grupos.to_numpy(),
            "real_bar": [BAR_POR_CLASE[c] for c in y],
            "pred_bar": [BAR_POR_CLASE[c] for c in prediccion],
            **{f"p_{BAR_POR_CLASE[c]}": probabilidades[:, i] for i, c in enumerate(CLASES)},
        })
        metricas.to_csv(destino / "metricas_clase.csv", index=False)
        predicciones.to_csv(destino / "predicciones.csv", index=False)
        _guardar_json(destino / "informe.json", informe)
        estado.update(estado="completada", fecha_fin_utc=_ahora())
        _guardar_json(registro, estado)
        return informe
    except Exception as error:
        estado.update(estado="fallida; consultar con el equipo antes de cualquier acción",
                      error=type(error).__name__, fecha_fin_utc=_ahora())
        _guardar_json(registro, estado)
        raise
