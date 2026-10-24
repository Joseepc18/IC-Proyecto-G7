"""Comprobaciones de entradas y de la app, usando únicamente entrenamiento."""
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from ic_g7 import config
from ic_g7.inferencia import preparar_aplicacion, predecir_ciclos
from ic_g7.prototipo import MAX_BYTES_CSV, detectar_fuera_de_rango, validar_csv


def mostrar_resultado(resultado):
    from ic_g7.interfaz.componentes import respuesta

    respuesta(resultado)


def mostrar_validacion(datos, error, fuera):
    from ic_g7.interfaz.componentes import validacion

    validacion(datos, error, fuera)


class PruebasPrototipo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modelo, cls.etiquetas, cls.X, cls.limites, cls.referencia = preparar_aplicacion()

    def csv(self, tabla):
        return tabla.to_csv(index=False).encode("utf-8")

    def test_ejemplos_de_todas_las_clases_y_limites_validos(self):
        # Un CSV exportado debe conservar números y orden al volver a cargarse.
        validos = pd.concat([
            self.X,
            self.limites["minimo"].to_frame().T,
            self.limites["maximo"].to_frame().T,
        ], ignore_index=True)
        resultado = validar_csv(self.csv(validos))
        self.assertEqual(resultado.shape, validos.shape)
        np.testing.assert_allclose(resultado, validos, rtol=1e-12, atol=1e-12)

    def test_columnas_desordenadas_se_normalizan_y_bom_se_acepta(self):
        tabla = self.X.head(1).iloc[:, ::-1]
        resultado = validar_csv(b"\xef\xbb\xbf" + self.csv(tabla))
        self.assertEqual(list(resultado.columns), config.COLUMNAS_ENTRADA)

    def test_columnas_faltantes_y_etiquetas_se_rechazan(self):
        base = self.X.head(1)
        casos = [
            base.drop(columns=base.columns[0]),
            base.assign(accumulator=130),
            base.assign(stable=1),
            base.assign(cycle_id=0),
        ]
        for tabla in casos:
            with self.subTest(columnas=tabla.columns.tolist()):
                with self.assertRaises(ValueError):
                    validar_csv(self.csv(tabla))

    def test_duplicados_y_cantidad_de_campos_se_rechazan(self):
        texto = self.csv(self.X.head(1)).decode()
        cabecera, fila = texto.splitlines()
        nombres = cabecera.split(",")
        nombres[-1] = nombres[0]
        casos = [
            ",".join(nombres) + "\n" + fila,
            cabecera + "\n" + fila + ",123",
            cabecera + "\n" + ",".join(fila.split(",")[:-1]),
        ]
        for caso in casos:
            with self.subTest(caso=caso[:50]):
                with self.assertRaises(ValueError):
                    validar_csv(caso.encode())

    def test_vacios_nulos_texto_e_infinitos_se_rechazan(self):
        for valor in ["", "NaN", "abc", "inf", "-inf"]:
            tabla = self.X.head(1).astype(object).copy()
            tabla.iloc[0, 0] = valor
            with self.subTest(valor=valor):
                with self.assertRaises(ValueError):
                    validar_csv(self.csv(tabla))

    def test_archivo_vacio_sin_registros_codificacion_y_tamano(self):
        casos = [
            b"", b"  ", self.csv(self.X.head(0)),
            b"\xff\xfe", b"x" * (MAX_BYTES_CSV + 1),
        ]
        for contenido in casos:
            with self.subTest(bytes=len(contenido)):
                with self.assertRaises(ValueError):
                    validar_csv(contenido)

    def test_salida_mapea_clases_a_bar_y_probabilidades(self):
        # Se comprueba el mapeo real de las cuatro clases, no una salida fija.
        predichas = self.modelo.predict(self.X)
        indices = [np.flatnonzero(predichas == c)[0] for c in self.modelo.classes_]
        entrada = self.X.iloc[indices]
        resultado = predecir_ciclos(self.modelo, self.etiquetas, entrada, self.limites, self.referencia)
        self.assertEqual(len(resultado), 4)
        self.assertEqual(resultado["Acumulador (bar)"].tolist(), [90, 100, 115, 130])
        np.testing.assert_allclose(
            resultado[[f"{b} bar (%)" for b in config.CLASE_POR_OBJETIVO]],
            self.modelo.predict_proba(entrada) * 100,
        )
        probas = resultado[["90 bar (%)", "100 bar (%)", "115 bar (%)", "130 bar (%)"]]
        np.testing.assert_allclose(probas.sum(axis=1), 100)

    def test_baja_confianza_segun_umbral(self):
        class ModeloFijo:
            classes_ = np.array([0, 1, 2, 3])

            def predict_proba(self, X):
                return np.array([
                    [0.40, 0.30, 0.20, 0.10],
                    [config.UMBRAL_CONFIANZA, 1 - config.UMBRAL_CONFIANZA, 0.0, 0.0],
                    [0.01, 0.01, 0.01, 0.97],
                ])[:len(X)]

            def predict(self, X):
                return self.classes_[self.predict_proba(X).argmax(axis=1)]

        resultado = predecir_ciclos(ModeloFijo(), self.etiquetas, self.X.head(3), self.limites, self.referencia)
        self.assertEqual(resultado["Baja confianza"].tolist(), [True, False, False])
        real = predecir_ciclos(self.modelo, self.etiquetas, self.X.head(3), self.limites, self.referencia)
        np.testing.assert_array_equal(
            real["Baja confianza"],
            self.modelo.predict_proba(self.X.head(3)).max(axis=1) < config.UMBRAL_CONFIANZA,
        )

    def test_interfaz_ejemplo_prediccion_y_cambio_de_origen(self):
        app = AppTest.from_file(str(config.RAIZ / "app" / "app.py")).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.radio[0].value, "Subir CSV")
        analizar = lambda: next(b for b in app.button if b.label == "Analizar datos")
        self.assertTrue(analizar().disabled)
        self.assertFalse(any(b.label == "Cargar ejemplo" for b in app.button))
        self.assertEqual(len(app.get("download_button")), 0)
        app.radio[0].set_value("Ciclo de ejemplo").run()
        self.assertEqual(len(app.get("download_button")), 1)
        next(b for b in app.button if b.label == "Cargar ejemplo").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.radio[0].value, "Ciclo de ejemplo")
        self.assertEqual(app.dataframe[0].value.shape, (1, 4))
        self.assertTrue(any(d.value.shape == (1, 68) for d in app.dataframe))
        self.assertFalse(analizar().disabled)
        analizar().click().run()
        self.assertEqual(len(app.exception), 0)
        resultado = app.session_state["resultado"]
        esperada = int(self.modelo.predict(self.X.head(1))[0])
        self.assertEqual(resultado.iloc[0]["Acumulador (bar)"], self.etiquetas[str(esperada)]["bar"])
        self.assertTrue(any(m.value == resultado.iloc[0]["Lectura de sensores"] for m in app.markdown))
        # La respuesta permanece al actualizar, pero desaparece al cambiar de entrada.
        app.run()
        self.assertIsNotNone(app.session_state["resultado"])
        self.assertTrue(any("Probabilidad" in d.value.columns for d in app.dataframe))
        app.radio[0].set_value("Subir CSV").run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(analizar().disabled)
        self.assertEqual(len(app.dataframe), 0)
        self.assertFalse(any(b.label == "Cargar ejemplo" for b in app.button))
        self.assertEqual(len(app.get("download_button")), 0)


    def test_carga_y_prediccion_sin_entrenar_ni_consultar_prueba(self):
        from sklearn.pipeline import Pipeline

        with (
            patch.object(Pipeline, "fit", side_effect=AssertionError("No entrenar en la app")),
            patch("ic_g7.datos.cargar_prueba", side_effect=AssertionError("No consultar prueba")),
        ):
            modelo, etiquetas, X, limites, referencia = preparar_aplicacion()
            with patch("ic_g7.inferencia.referencia_desde_entrenamiento",
                       side_effect=AssertionError("No recalcular la referencia al predecir")):
                resultado = predecir_ciclos(modelo, etiquetas, X.head(10), limites, referencia)
        self.assertEqual(len(resultado), 10)
        np.testing.assert_array_equal(resultado["Registro"], np.arange(1, 11))
        self.assertTrue(resultado["Acumulador (bar)"].isin([90, 100, 115, 130]).all())

    def test_modelo_ausente_detiene_la_app_con_error_claro(self):
        with TemporaryDirectory() as carpeta, patch.object(config, "DIR_MODELOS", Path(carpeta)):
            app = AppTest.from_file(str(config.RAIZ / "app" / "app.py")).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("modelo final" in e.value for e in app.error))
        self.assertFalse(any(b.label == "Analizar datos" for b in app.button))

    def test_interfaz_muestra_un_resultado_por_ciclo_sin_tablas_duplicadas(self):
        # El CSV subido pasa por la misma validación y predicción del ejemplo.
        entrada = validar_csv(self.csv(self.X.head(10)))
        resultado = predecir_ciclos(self.modelo, self.etiquetas, entrada, self.limites, self.referencia)
        app = AppTest.from_function(mostrar_resultado, args=(resultado,)).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual(app.dataframe[0].value.shape, (10, 10))
        explicacion = app.dataframe[0].value.iloc[0]["Lectura de sensores"]
        self.assertTrue(any(m.value == explicacion for m in app.markdown))
        app.selectbox[0].select(10).run()
        self.assertTrue(any(m.value == resultado.iloc[9]["Lectura de sensores"] for m in app.markdown))
        self.assertEqual(len(app.get("download_button")), 1)


    def test_fuera_de_rango_se_admite_sin_recortar_y_se_marca_por_ciclo(self):
        tabla = self.X.head(2).copy()
        tabla.iloc[1, 0] = self.limites.iloc[0]["maximo"] + 10
        entrada = validar_csv(self.csv(tabla))
        np.testing.assert_allclose(entrada, tabla)
        fuera = detectar_fuera_de_rango(entrada, self.limites)
        self.assertEqual(fuera.any(axis=1).tolist(), [False, True])
        resultado = predecir_ciclos(
            self.modelo, self.etiquetas, entrada, self.limites, self.referencia,
        )
        self.assertEqual(resultado["Fuera de rango"].tolist(), [False, True])
        np.testing.assert_allclose(
            resultado[[f"{b} bar (%)" for b in config.CLASE_POR_OBJETIVO]],
            self.modelo.predict_proba(entrada) * 100,
        )
        app = AppTest.from_function(
            mostrar_validacion, args=(entrada, None, fuera),
        ).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.warning), 1)
        self.assertIn("1 de 2", app.warning[0].value)
        self.assertEqual(len(app.error), 0)

    def test_redondeo_del_limite_no_produce_aviso(self):
        tabla = self.X.head(1).copy()
        maximo = self.limites.iloc[0]["maximo"]
        tabla.iloc[0, 0] = maximo + 0.5e-10 * abs(maximo)
        self.assertFalse(detectar_fuera_de_rango(tabla, self.limites).to_numpy().any())
        tabla.iloc[0, 0] = maximo + 0.001
        self.assertTrue(detectar_fuera_de_rango(tabla, self.limites).iloc[0, 0])

    def test_explicacion_conserva_orden_y_recibe_la_clase_predicha(self):
        from ic_g7.difuso import explicar

        entrada = self.X.iloc[[8, 1, 6]]
        resultado = predecir_ciclos(
            self.modelo, self.etiquetas, entrada, self.limites, self.referencia,
        )
        clases = self.modelo.predict(entrada)
        esperadas = [
            explicar(entrada.iloc[i], self.referencia, clase_predicha=int(clase))
            for i, clase in enumerate(clases)
        ]
        self.assertEqual(resultado["Lectura de sensores"].tolist(), esperadas)
        self.assertEqual(resultado["Registro"].tolist(), [1, 2, 3])

    def test_discrepancia_difusa_avisa_sin_cambiar_prediccion(self):
        entrada = self.X.head(1).copy()
        for variable, percentiles in self.referencia.items():
            entrada.loc[0, variable] = percentiles[0]
        modelo = Mock(classes_=np.array([0, 1, 2, 3]))
        modelo.predict_proba.return_value = np.array([[1.0, 0.0, 0.0, 0.0]])
        resultado = predecir_ciclos(
            modelo, self.etiquetas, entrada, self.limites, self.referencia,
        )
        self.assertEqual(resultado.iloc[0]["Acumulador (bar)"], 90)
        self.assertIn("no coincide", resultado.iloc[0]["Lectura de sensores"])

    def test_fuera_de_rango_no_se_confunde_con_baja_confianza(self):
        entrada = self.X.head(1).copy()
        entrada.iloc[0, 0] = self.limites.iloc[0]["maximo"] + 10
        modelo = Mock(classes_=np.array([0, 1, 2, 3]))
        modelo.predict_proba.return_value = np.array([[0.0, 0.0, 0.0, 1.0]])
        resultado = predecir_ciclos(
            modelo, self.etiquetas, entrada, self.limites, self.referencia,
        )
        self.assertTrue(resultado.iloc[0]["Fuera de rango"])
        self.assertFalse(resultado.iloc[0]["Baja confianza"])


if __name__ == "__main__":
    unittest.main()
