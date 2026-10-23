"""Experimentos de Random Forest y MLP de Dayron, exclusivamente con entrenamiento.

Reutiliza el Pipeline y la evaluación del equipo. No selecciona ni guarda el
modelo final; genera evidencia de validación para la reunión del equipo.
"""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, cross_val_predict
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_info, threadpool_limits

from ic_g7 import config
from ic_g7.datos import cargar_entrenamiento
from ic_g7.evaluacion import crear_cv, validar_cruzado
from ic_g7.pipeline import crear_pipeline

# José comprobó este límite en 03_metaheuristica; los demás valores son los de sklearn.
MLP_MAX_ITER = 4000
NOMBRE_RF_BASE = "Random Forest por defecto"
NOMBRE_RF_AJUSTADO = "Random Forest ajustado"
NOMBRE_MLP = "MLP por defecto (Dayron)"
BAR_POR_CLASE = {clase: bar for bar, clase in config.CLASE_POR_OBJETIVO.items()}
CLASES = np.array(sorted(BAR_POR_CLASE))
BARRAS = [BAR_POR_CLASE[c] for c in CLASES]
REJILLA_RF = {
    "modelo__n_estimators": [100, 200, 300],
    "modelo__max_depth": [None, 10, 20],
    "modelo__min_samples_leaf": [1, 2, 4],
}


def crear_modelos(parametros_rf):
    """La búsqueda solo modifica RF; la MLP conserva sus valores por defecto."""
    rf_base = crear_pipeline(RandomForestClassifier(
        random_state=config.RANDOM_STATE, n_jobs=1,
    ))
    rf_ajustado = crear_pipeline(RandomForestClassifier(
        random_state=config.RANDOM_STATE, n_jobs=1,
    )).set_params(**parametros_rf)
    mlp = crear_pipeline(MLPClassifier(
        max_iter=MLP_MAX_ITER, random_state=config.RANDOM_STATE,
    ))
    return {
        NOMBRE_RF_BASE: rf_base,
        NOMBRE_RF_AJUSTADO: rf_ajustado,
        NOMBRE_MLP: mlp,
    }


def comprobar_pliegues(X, y, grupos, pliegues):
    """Impide mezclar grupos o dejar filas sin una única predicción de validación."""
    if list(X.columns) != config.COLUMNAS_ENTRADA:
        raise ValueError("Se requieren las 68 entradas, sin etiquetas ni identificadores.")
    if not (len(X) == len(y) == len(grupos)):
        raise ValueError("Entradas, etiquetas y grupos deben tener la misma longitud.")
    if len(pliegues) != config.N_PLIEGUES:
        raise ValueError("Se requieren los cinco pliegues del contrato.")
    vistos = np.zeros(len(X), dtype=int)
    for entrenamiento, validacion in pliegues:
        if set(grupos.iloc[entrenamiento]) & set(grupos.iloc[validacion]):
            raise ValueError("Hay grupos compartidos entre entrenamiento y validación.")
        if set(entrenamiento) & set(validacion):
            raise ValueError("Hay filas compartidas entre entrenamiento y validación.")
        if len(entrenamiento) + len(validacion) != len(X):
            raise ValueError("Cada pliegue debe partir todo el entrenamiento disponible.")
        for indices in (entrenamiento, validacion):
            if not np.array_equal(np.unique(y.iloc[indices]), CLASES):
                raise ValueError("Cada partición del pliegue debe contener las cuatro clases.")
        vistos[validacion] += 1
    if not np.all(vistos == 1):
        raise ValueError("Cada ciclo debe validarse exactamente una vez.")


def resumir_oof(nombre, y, probabilidades, grupos, pliegues):
    """Resume predicciones de filas excluidas del ajuste de su propio pliegue."""
    if probabilidades.shape != (len(y), len(CLASES)):
        raise ValueError("Las probabilidades deben tener una fila por ciclo y cuatro clases.")
    if (not np.isfinite(probabilidades).all()
            or (probabilidades < 0).any() or (probabilidades > 1).any()
            or not np.allclose(probabilidades.sum(axis=1), 1)):
        raise ValueError("Las probabilidades deben ser finitas y sumar uno por ciclo.")
    prediccion = CLASES[probabilidades.argmax(axis=1)]
    precision, recall, f1, soporte = precision_recall_fscore_support(
        y, prediccion, labels=CLASES, zero_division=0,
    )
    metricas = pd.DataFrame({
        "modelo": nombre, "clase_bar": BARRAS,
        "precision": precision, "recall": recall, "f1": f1, "soporte": soporte,
    })
    matriz = confusion_matrix(y, prediccion, labels=CLASES)
    matrices = pd.DataFrame(matriz, columns=[f"pred_{bar}" for bar in BARRAS])
    matrices.insert(0, "real_bar", BARRAS)
    matrices.insert(0, "modelo", nombre)
    numero_pliegue = np.full(len(y), -1, dtype=int)
    for numero, (_, indices) in enumerate(pliegues):
        numero_pliegue[indices] = numero
    predicciones = pd.DataFrame({
        "modelo": nombre, "indice_entrenamiento": np.arange(len(y)),
        "group_id": grupos.to_numpy(), "pliegue": numero_pliegue,
        "real_bar": [BAR_POR_CLASE[c] for c in y],
        "pred_bar": [BAR_POR_CLASE[c] for c in prediccion],
        **{f"p_{bar}": probabilidades[:, i] for i, bar in enumerate(BARRAS)},
    })
    return metricas, matrices, predicciones


def evaluar_modelo(nombre, modelo, X, y, grupos, pliegues):
    """Obtiene la tabla común y comprueba que sus métricas coincidan con las OOF."""
    parametros = {
        **modelo.named_steps["modelo"].get_params(),
        "umbral_correlacion": modelo.named_steps["correlacion"].umbral,
    }
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        fila, por_pliegue = validar_cruzado(
            modelo, X, y, grupos, nombre, hiperparametros=parametros,
        )
        probabilidades = cross_val_predict(
            modelo, X, y, groups=grupos, cv=pliegues,
            method="predict_proba", n_jobs=1,
        )
    convergencia = [a for a in avisos if issubclass(a.category, ConvergenceWarning)]
    if convergencia:
        raise RuntimeError(f"{nombre}: hubo {len(convergencia)} avisos de convergencia.")
    if avisos:
        # No ocultar advertencias nuevas de bibliotecas o datos.
        for aviso in avisos:
            warnings.warn(str(aviso.message), aviso.category, stacklevel=2)

    prediccion = CLASES[probabilidades.argmax(axis=1)]
    for numero, (_, indices) in enumerate(pliegues):
        medidas = [
            f1_score(y.iloc[indices], prediccion[indices], average="macro"),
            accuracy_score(y.iloc[indices], prediccion[indices]),
            roc_auc_score(y.iloc[indices], probabilidades[indices], multi_class="ovr"),
        ]
        np.testing.assert_allclose(
            por_pliegue.loc[numero, ["f1_macro", "exactitud", "roc_auc"]].to_numpy(float),
            medidas, rtol=1e-10, atol=1e-10,
        )
    metricas, matrices, predicciones = resumir_oof(
        nombre, y, probabilidades, grupos, pliegues,
    )
    return fila, por_pliegue.assign(modelo=nombre), metricas, matrices, predicciones


def guardar_figura(matrices, ruta):
    """Una matriz por modelo, con filas reales y columnas predichas, en bar."""
    nombres = matrices["modelo"].drop_duplicates().tolist()
    figura, ejes = plt.subplots(1, len(nombres), figsize=(15, 4.5), squeeze=False)
    for eje, nombre in zip(ejes[0], nombres):
        valores = matrices.loc[matrices["modelo"] == nombre, [f"pred_{b}" for b in BARRAS]]
        ConfusionMatrixDisplay(valores.to_numpy(), display_labels=BARRAS).plot(
            ax=eje, colorbar=False, cmap="Blues", values_format="d",
        )
        eje.set_title(nombre, fontsize=11)
        eje.set_xlabel("Categoría predicha (bar)")
        eje.set_ylabel("Categoría real (bar)")
    figura.suptitle("Validación fuera de pliegue · solo entrenamiento", fontsize=13)
    figura.tight_layout()
    figura.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(figura)


def ejecutar_experimentos(directorio=config.DIR_REPORTES):
    """Ejecuta RF y MLP, exporta solo archivos de Dayron y devuelve sus tablas."""
    directorio = Path(directorio)
    X, y, grupos = cargar_entrenamiento()
    pliegues = list(crear_cv().split(X, y, grupos))
    comprobar_pliegues(X, y, grupos, pliegues)
    hash_split = sha256(config.RUTA_SPLIT.read_bytes()).hexdigest()
    inicio_total = perf_counter()
    print(f"Solo entrenamiento: {len(X)} ciclos, {X.shape[1]} variables, "
          f"{grupos.nunique()} grupos.", flush=True)
    print(f"RF: {len(ParameterGrid(REJILLA_RF))} candidatos x {len(pliegues)} pliegues.",
          flush=True)
    filas, tablas_pliegues, metricas, matrices, predicciones = [], [], [], [], []

    # Un proceso y un hilo BLAS evitan competencia entre experimentos al medir tiempos.
    with threadpool_limits(limits=1):
        bibliotecas_numericas = threadpool_info()
        busqueda = GridSearchCV(
            crear_pipeline(RandomForestClassifier(
                random_state=config.RANDOM_STATE, n_jobs=1,
            )),
            REJILLA_RF, scoring="f1_macro", cv=pliegues, n_jobs=1,
            refit=False, error_score="raise", return_train_score=False, verbose=1,
        )
        inicio_busqueda = perf_counter()
        busqueda.fit(X, y, groups=grupos)
        segundos_busqueda = perf_counter() - inicio_busqueda
        print(f"RF: búsqueda terminada en {segundos_busqueda:.1f} s; "
              f"mejor F1 CV = {busqueda.best_score_:.4f}.", flush=True)
        for nombre, modelo in crear_modelos(busqueda.best_params_).items():
            print(f"Evaluando {nombre}...", flush=True)
            fila, tabla, metrica, matriz, prediccion = evaluar_modelo(
                nombre, modelo, X, y, grupos, pliegues,
            )
            filas.append(fila)
            tablas_pliegues.append(tabla)
            metricas.append(metrica)
            matrices.append(matriz)
            predicciones.append(prediccion)
            print(f"  F1 macro: {fila['F1_macro_CV (media±std)']}", flush=True)

    if sha256(config.RUTA_SPLIT.read_bytes()).hexdigest() != hash_split:
        raise RuntimeError("La partición compartida cambió durante el experimento.")
    tablas = {
        "resultados": pd.DataFrame(filas, columns=config.COLUMNAS_RESULTADOS),
        "pliegues": pd.concat(tablas_pliegues, ignore_index=True),
        "metricas_clase": pd.concat(metricas, ignore_index=True),
        "matrices_confusion": pd.concat(matrices, ignore_index=True),
        "predicciones_oof": pd.concat(predicciones, ignore_index=True),
        "busqueda_rf": pd.DataFrame(busqueda.cv_results_),
    }
    fila_rf = tablas["pliegues"].query("modelo == @NOMBRE_RF_AJUSTADO")
    np.testing.assert_allclose(fila_rf["f1_macro"].mean(), busqueda.best_score_, atol=1e-10)

    entorno = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "sistema": platform.platform(), "procesador": platform.processor(),
        "cpu_logicos": psutil.cpu_count(logical=True),
        "ram_gib": round(psutil.virtual_memory().total / 1024**3, 2),
        "python": platform.python_version(),
        "versiones": {p: version(p) for p in (
            "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "psutil", "threadpoolctl",
        )},
        "semilla": config.RANDOM_STATE, "mlp_max_iter": MLP_MAX_ITER,
        "pliegues": len(pliegues), "ciclos_entrenamiento": len(X),
        "variables_entrada": X.shape[1], "grupos_entrenamiento": int(grupos.nunique()),
        "clases_entrenamiento": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "split_sha256": hash_split,
        "features_sha256": sha256(config.RUTA_FEATURES.read_bytes()).hexdigest(),
        "n_jobs": 1, "limite_hilos_blas": 1,
        "bibliotecas_numericas": [
            {k: b.get(k) for k in ("internal_api", "version", "num_threads", "threading_layer")}
            for b in bibliotecas_numericas
        ],
        "candidatos_rf": len(ParameterGrid(REJILLA_RF)),
        "ajustes_busqueda_rf": len(ParameterGrid(REJILLA_RF)) * len(pliegues),
        "mejores_parametros_rf": busqueda.best_params_,
        "f1_cv_busqueda_rf": float(busqueda.best_score_),
        "segundos_busqueda_rf": segundos_busqueda,
        "segundos_experimento": perf_counter() - inicio_total,
        "avisos_convergencia": 0, "conjunto_prueba_evaluado": False,
        "alcance": "CV de entrenamiento; la búsqueda y la comparación reutilizan los pliegues. "
                   "Las OOF del RF ajustado no son una evaluación independiente de su selección. "
                   "Los tiempos por modelo no incluyen la búsqueda ni la segunda pasada OOF.",
    }
    directorio.mkdir(parents=True, exist_ok=True)
    for nombre, tabla in tablas.items():
        tabla.to_csv(directorio / f"{nombre}_dayron.csv", index=False)
    (directorio / "entorno_dayron.json").write_text(
        json.dumps(entorno, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    figuras = directorio / "figuras"
    figuras.mkdir(exist_ok=True)
    guardar_figura(tablas["matrices_confusion"], figuras / "matrices_confusion_dayron.png")
    print("Archivos de Dayron guardados; la tabla consolidada no se modifica.", flush=True)
    return tablas
