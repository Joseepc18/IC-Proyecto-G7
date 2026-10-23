"""Preparación solo con train, artefactos íntegros y evaluación única con datos sintéticos."""
import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

from ic_g7 import config
from ic_g7.modelo_final import (
    ARCHIVOS_MODELO, PARAMETROS_SELECCIONADOS, cargar_modelo_guardado,
    crear_modelo_elegido, evaluar_prueba_una_vez, leer_seleccion, preparar_modelo,
)
from ic_g7.pipeline import crear_pipeline


class PruebasModeloFinal(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raiz = Path(self.tmp.name)
        self.modelos = self.raiz / "models"
        self.reportes = self.raiz / "reports"
        self.reportes.mkdir()
        self.evaluacion = self.raiz / "evaluacion"
        self.split = self.raiz / "split.csv"
        self.split.write_text("partición artificial", encoding="utf-8")
        pd.DataFrame([{
            "modelo": "MLP+GA (semilla 42)",
            "hiperparámetros": repr(PARAMETROS_SELECCIONADOS),
            "F1_macro_CV (media±std)": "0.8123 ± 0.0543",
        }]).to_csv(self.reportes / "resultados_jose.csv", index=False)
        rng = np.random.default_rng(config.RANDOM_STATE)
        self.X = pd.DataFrame(rng.normal(size=(40, 68)), columns=config.COLUMNAS_ENTRADA)
        self.y = pd.Series(np.tile(np.arange(4), 10))
        self.grupos = pd.Series(np.repeat(np.arange(10), 4))
        self.X_test = self.X.iloc[:8].copy().reset_index(drop=True)
        self.y_test = self.y.iloc[:8].copy().reset_index(drop=True)
        self.g_test = pd.Series(np.repeat([100, 101], 4))
        for objetivo, valor in (
            ("ic_g7.modelo_final.cargar_entrenamiento", (self.X, self.y, self.grupos)),
        ):
            parche = patch(objetivo, return_value=valor)
            parche.start()
            self.addCleanup(parche.stop)
        for campo, valor in (("DIR_REPORTES", self.reportes), ("RUTA_SPLIT", self.split)):
            parche = patch.object(config, campo, valor)
            parche.start()
            self.addCleanup(parche.stop)

    def _preparar(self):
        with patch("ic_g7.datos.cargar_prueba", side_effect=AssertionError("Prueba prohibida")):
            return preparar_modelo(self.modelos)

    def test_guarda_pipeline_completo_y_etiquetas_sin_cargar_prueba(self):
        metadata = self._preparar()
        modelo, etiquetas, recuperada = cargar_modelo_guardado(self.modelos)
        self.assertEqual(set(ARCHIVOS_MODELO), {p.name for p in self.modelos.iterdir()})
        self.assertEqual(list(modelo.named_steps), ["escalador", "correlacion", "modelo"])
        for parametro, valor in PARAMETROS_SELECCIONADOS.items():
            self.assertEqual(modelo.named_steps["modelo"].get_params()[parametro], valor)
        self.assertEqual(etiquetas["0"], {"bar": 90, "estado": "falla_cercana"})
        self.assertEqual(etiquetas["3"], {"bar": 130, "estado": "optima"})
        self.assertEqual(metadata["ciclos_entrenamiento"], 40)
        self.assertFalse(metadata["prueba_usada_en_reentrenamiento"])
        self.assertEqual(metadata, recuperada)
        referencia = crear_modelo_elegido().fit(self.X, self.y)
        np.testing.assert_array_equal(modelo.predict(self.X), referencia.predict(self.X))
        np.testing.assert_allclose(modelo.predict_proba(self.X), referencia.predict_proba(self.X))
        self.assertFalse(self.evaluacion.exists())

    def test_no_sobrescribe_artefactos_ni_reentrena_por_accidente(self):
        self._preparar()
        contenido = {p.name: p.read_bytes() for p in self.modelos.iterdir()}
        with patch("sklearn.neural_network.MLPClassifier.fit", side_effect=AssertionError("No ajustar")):
            with self.assertRaises(FileExistsError):
                preparar_modelo(self.modelos)
        self.assertEqual(contenido, {p.name: p.read_bytes() for p in self.modelos.iterdir()})

    def test_no_exporta_si_no_converge(self):
        modelo = crear_pipeline(MLPClassifier(
            hidden_layer_sizes=(64,), max_iter=1, random_state=config.RANDOM_STATE,
        ))
        with patch("ic_g7.modelo_final.crear_modelo_elegido", return_value=modelo):
            with self.assertRaisesRegex(RuntimeError, "no convergió"):
                preparar_modelo(self.modelos)
        self.assertFalse(self.modelos.exists())

    def test_rechaza_seleccion_distinta_y_archivo_modificado(self):
        tabla = pd.read_csv(self.reportes / "resultados_jose.csv")
        tabla["hiperparámetros"] = repr({**PARAMETROS_SELECCIONADOS, "alpha": 0.1})
        tabla.to_csv(self.reportes / "resultados_jose.csv", index=False)
        with self.assertRaisesRegex(ValueError, "configuración registrada"):
            leer_seleccion()

    def test_rechaza_artefacto_alterado_antes_de_deserializar(self):
        self._preparar()
        with (self.modelos / "pipeline_final.joblib").open("ab") as archivo:
            archivo.write(b"alterado")
        with patch("ic_g7.modelo_final.joblib.load", side_effect=AssertionError("No cargar")):
            with self.assertRaisesRegex(ValueError, "huella"):
                cargar_modelo_guardado(self.modelos)

    def test_sin_confirmacion_no_carga_modelo_ni_prueba(self):
        with (
            patch("ic_g7.modelo_final.cargar_modelo_guardado",
                  side_effect=AssertionError("No cargar modelo")),
            patch("ic_g7.datos.cargar_prueba", side_effect=AssertionError("No cargar prueba")),
            self.assertRaises(PermissionError),
        ):
            evaluar_prueba_una_vez(
                directorio_modelos=self.modelos, directorio_reportes=self.evaluacion,
            )
        self.assertFalse(self.evaluacion.exists())

    def test_cli_exige_confirmacion_de_evaluacion_sin_reunion(self):
        ruta = Path(__file__).resolve().parents[1] / "scripts" / "evaluar_modelo_final.py"
        spec = importlib.util.spec_from_file_location("evaluar_final_cli", ruta)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with (
            patch.object(cli, "evaluar_prueba_una_vez") as evaluar,
            patch.object(sys, "argv", [str(ruta)]),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            cli.main()
        self.assertEqual(error.exception.code, 2)
        evaluar.assert_not_called()
        with (
            patch.object(cli, "evaluar_prueba_una_vez", return_value={
                "f1_macro_prueba": 0.8, "exactitud_prueba": 0.8,
                "roc_auc_ovr_macro_prueba": 0.9,
            }) as evaluar,
            patch.object(sys, "argv", [str(ruta), "--confirmar-evaluacion"]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            cli.main()
        evaluar.assert_called_once_with(confirmar_evaluacion=True)

    def test_evalua_una_vez_sin_fit_y_mantiene_artefactos(self):
        self._preparar()
        antes = {p.name: p.read_bytes() for p in self.modelos.iterdir()}
        with (
            patch("ic_g7.datos.cargar_prueba",
                  return_value=(self.X_test, self.y_test, self.g_test)) as cargar,
            patch("sklearn.neural_network.MLPClassifier.fit",
                  side_effect=AssertionError("No ajustar con prueba")),
        ):
            informe = evaluar_prueba_una_vez(True, self.modelos, self.evaluacion)
            cargar.assert_called_once_with()
            with self.assertRaises(FileExistsError):
                evaluar_prueba_una_vez(True, self.modelos, self.evaluacion)
            cargar.assert_called_once_with()
        estado = json.loads((self.evaluacion / "registro.json").read_text(encoding="utf-8"))
        self.assertEqual(estado["estado"], "completada")
        self.assertIn("evaluación única acordada", estado["confirmacion"])
        self.assertNotIn("reunión", estado["confirmacion"])
        self.assertEqual(informe["ciclos_prueba"], len(self.X_test))
        self.assertEqual(informe["confianza"]["umbral"], config.UMBRAL_CONFIANZA)
        self.assertEqual(antes, {p.name: p.read_bytes() for p in self.modelos.iterdir()})
        modelo = joblib.load(self.modelos / "pipeline_final.joblib")
        proba = modelo.predict_proba(self.X_test)
        pred = modelo.predict(self.X_test)
        self.assertAlmostEqual(informe["f1_macro_prueba"], f1_score(self.y_test, pred, average="macro"))
        self.assertAlmostEqual(informe["exactitud_prueba"], accuracy_score(self.y_test, pred))
        self.assertAlmostEqual(
            informe["roc_auc_ovr_macro_prueba"],
            roc_auc_score(self.y_test, proba, multi_class="ovr"),
        )
        self.assertEqual(np.array(informe["matriz_confusion"]).sum(), len(self.X_test))
        self.assertEqual(informe["criterio_b"]["cumple"],
                         informe["f1_macro_prueba"] >= 0.8123 - 2 * 0.0543)

    def test_datos_cambiados_no_permiten_consultar_prueba(self):
        self._preparar()
        self.X.iloc[0, 0] += 1
        with (
            patch("ic_g7.datos.cargar_prueba", side_effect=AssertionError("No cargar prueba")),
            self.assertRaisesRegex(ValueError, "datos o la partición cambiaron"),
        ):
            evaluar_prueba_una_vez(True, self.modelos, self.evaluacion)
        self.assertFalse(self.evaluacion.exists())

    def test_rechaza_grupos_compartidos_y_no_reintenta_tras_fallo(self):
        self._preparar()
        with patch("ic_g7.datos.cargar_prueba",
                   return_value=(self.X_test, self.y_test, self.grupos.iloc[:8])) as cargar:
            with self.assertRaisesRegex(ValueError, "grupos compartidos"):
                evaluar_prueba_una_vez(True, self.modelos, self.evaluacion)
            with self.assertRaises(FileExistsError):
                evaluar_prueba_una_vez(True, self.modelos, self.evaluacion)
            cargar.assert_called_once_with()
        registro = json.loads((self.evaluacion / "registro.json").read_text(encoding="utf-8"))
        self.assertTrue(registro["estado"].startswith("fallida"))
        self.assertFalse((self.evaluacion / "informe.json").exists())


if __name__ == "__main__":
    unittest.main()
