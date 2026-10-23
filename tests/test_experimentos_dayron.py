"""Controles de aislamiento por grupos, salidas OOF y ejecución solo con train."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning

from ic_g7 import config
from ic_g7.evaluacion import crear_cv
from ic_g7.experimentos_dayron import (
    NOMBRE_MLP, NOMBRE_RF_AJUSTADO, comprobar_pliegues, crear_modelos,
    ejecutar_experimentos, evaluar_modelo, resumir_oof,
)
from ic_g7.pipeline import crear_pipeline


class PruebasExperimentosDayron(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(config.RANDOM_STATE)
        self.X = pd.DataFrame(rng.normal(size=(40, 68)), columns=config.COLUMNAS_ENTRADA)
        self.y = pd.Series(np.tile(np.arange(4), 10))
        self.grupos = pd.Series(np.repeat(np.arange(10), 4))
        self.pliegues = list(crear_cv().split(self.X, self.y, self.grupos))

    def test_rechaza_fuga_de_grupos_y_ciclos_sin_validar(self):
        comprobar_pliegues(self.X, self.y, self.grupos, self.pliegues)
        # Partir ciclos alternos de cada grupo mantiene las clases, pero filtra grupos.
        fuga = [(np.arange(0, 40, 2), np.arange(1, 40, 2))] * config.N_PLIEGUES
        with self.assertRaisesRegex(ValueError, "grupos compartidos"):
            comprobar_pliegues(self.X, self.y, self.grupos, fuga)
        repetidos = [self.pliegues[0]] * config.N_PLIEGUES
        with self.assertRaisesRegex(ValueError, "exactamente una vez"):
            comprobar_pliegues(self.X, self.y, self.grupos, repetidos)

    def test_rechaza_etiquetas_como_entradas(self):
        with self.assertRaisesRegex(ValueError, "68 entradas"):
            comprobar_pliegues(
                self.X.assign(accumulator=130), self.y, self.grupos, self.pliegues,
            )

    def test_oof_conserva_categorias_probabilidades_y_soportes(self):
        probabilidades = np.eye(4)[self.y]
        # Un caso real de 90 se clasifica como 130: comprueba filas y columnas en bar.
        probabilidades[0] = [0, 0, 0, 1]
        metricas, matrices, predicciones = resumir_oof(
            "prueba", self.y, probabilidades, self.grupos, self.pliegues,
        )
        self.assertEqual(matrices.set_index("real_bar").loc[90, "pred_130"], 1)
        self.assertEqual(matrices.set_index("real_bar").loc[130, "pred_90"], 0)
        self.assertEqual(metricas.set_index("clase_bar").loc[90, "recall"], 0.9)
        self.assertEqual(metricas["soporte"].sum(), len(self.y))
        self.assertEqual(predicciones.iloc[0]["real_bar"], 90)
        self.assertEqual(predicciones.iloc[0]["pred_bar"], 130)
        self.assertEqual(predicciones.iloc[0]["p_130"], 1)
        for _, tabla in predicciones.groupby("group_id"):
            self.assertEqual(tabla["pliegue"].nunique(), 1)

    def test_no_admite_probabilidades_invalidas(self):
        for mala in (
            np.zeros((40, 4)), np.full((40, 4), np.nan),
            np.tile([1.1, -0.1, 0, 0], (40, 1)),
        ):
            with self.subTest(primera_fila=mala[0]):
                with self.assertRaisesRegex(ValueError, "probabilidades"):
                    resumir_oof("prueba", self.y, mala, self.grupos, self.pliegues)

    def test_detiene_experimento_si_mlp_no_converge(self):
        modelo = crear_modelos({})[NOMBRE_MLP]
        probabilidades = np.eye(4)[self.y]
        por_pliegue = pd.DataFrame({
            "pliegue": range(config.N_PLIEGUES),
            "f1_macro": 1.0, "exactitud": 1.0, "roc_auc": 1.0,
        })
        # Si se quitara el control, estas salidas válidas permitirían continuar.
        for origen in ("validacion", "oof"):
            with (
                self.subTest(origen=origen),
                patch("ic_g7.experimentos_dayron.validar_cruzado",
                      return_value=({}, por_pliegue)) as validar,
                patch("ic_g7.experimentos_dayron.cross_val_predict",
                      return_value=probabilidades) as predecir,
            ):
                llamada, salida = (
                    (validar, ({}, por_pliegue)) if origen == "validacion"
                    else (predecir, probabilidades)
                )

                def avisar_no_convergencia(*args, **kwargs):
                    warnings.warn("La MLP no terminó de converger.", ConvergenceWarning)
                    return salida

                llamada.side_effect = avisar_no_convergencia
                with self.assertRaisesRegex(
                    RuntimeError, "MLP por defecto.*1 avisos de convergencia",
                ):
                    evaluar_modelo(
                        NOMBRE_MLP, modelo, self.X, self.y, self.grupos, self.pliegues,
                    )

    def test_ejecucion_exporta_solo_dayron_y_no_consulta_prueba(self):
        # La prueba usa una rejilla pequeña y datos artificiales; la ejecución real usa 27.
        rejilla = {"modelo__n_estimators": [3]}
        modelo = crear_pipeline(RandomForestClassifier(
            n_estimators=3, n_jobs=1, random_state=config.RANDOM_STATE,
        ))
        with TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            split, features = raiz / "split.csv", raiz / "features.csv"
            split.write_text("particion de prueba unitaria", encoding="utf-8")
            features.write_text("datos artificiales", encoding="utf-8")
            reportes = raiz / "reports"
            reportes.mkdir()
            compartido = reportes / "resultados.csv"
            compartido.write_text("conservar", encoding="utf-8")
            with (
                patch("ic_g7.experimentos_dayron.cargar_entrenamiento",
                      return_value=(self.X, self.y, self.grupos)),
                patch("ic_g7.datos.cargar_prueba", side_effect=AssertionError("Prueba prohibida")),
                patch("ic_g7.experimentos_dayron.REJILLA_RF", rejilla),
                patch("ic_g7.experimentos_dayron.crear_modelos",
                      return_value={NOMBRE_RF_AJUSTADO: modelo}),
                patch.object(config, "RUTA_SPLIT", split),
                patch.object(config, "RUTA_FEATURES", features),
            ):
                tablas = ejecutar_experimentos(reportes)
            self.assertEqual(compartido.read_text(encoding="utf-8"), "conservar")
            self.assertEqual(split.read_text(encoding="utf-8"), "particion de prueba unitaria")
            self.assertEqual(len(tablas["pliegues"]), config.N_PLIEGUES)
            self.assertEqual(len(tablas["predicciones_oof"]), len(self.X))
            self.assertEqual(tablas["resultados"].columns.tolist(), config.COLUMNAS_RESULTADOS)
            entorno = json.loads((reportes / "entorno_dayron.json").read_text(encoding="utf-8"))
            self.assertFalse(entorno["conjunto_prueba_evaluado"])
            self.assertEqual(entorno["avisos_convergencia"], 0)
            self.assertTrue((reportes / "figuras/matrices_confusion_dayron.png").is_file())


if __name__ == "__main__":
    unittest.main()
