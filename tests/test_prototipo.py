"""Comprobaciones de entradas y de la app, usando únicamente entrenamiento."""
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from ic_g7 import config
from ic_g7.prototipo import (
    MAX_BYTES_CSV,
    predecir_provisional,
    preparar_prototipo,
    validar_csv,
)


class PruebasPrototipo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modelo, cls.X, cls.limites = preparar_prototipo()

    def csv(self, tabla):
        return tabla.to_csv(index=False).encode("utf-8")

    def test_ejemplos_de_todas_las_clases_y_limites_validos(self):
        # Un CSV exportado debe conservar números y orden al volver a cargarse.
        validos = pd.concat([
            self.X,
            self.limites["minimo"].to_frame().T,
            self.limites["maximo"].to_frame().T,
        ], ignore_index=True)
        resultado = validar_csv(self.csv(validos), self.limites)
        self.assertEqual(resultado.shape, validos.shape)
        np.testing.assert_allclose(resultado, validos, rtol=1e-12, atol=1e-12)

    def test_columnas_desordenadas_se_normalizan_y_bom_se_acepta(self):
        tabla = self.X.head(1).iloc[:, ::-1]
        resultado = validar_csv(b"\xef\xbb\xbf" + self.csv(tabla), self.limites)
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
                    validar_csv(self.csv(tabla), self.limites)

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
                    validar_csv(caso.encode(), self.limites)

    def test_vacios_nulos_texto_infinitos_y_fuera_de_rango(self):
        for valor in ["", "NaN", "abc", "inf", "-inf",
                      self.limites.iloc[0]["maximo"] + 10]:
            tabla = self.X.head(1).astype(object).copy()
            tabla.iloc[0, 0] = valor
            with self.subTest(valor=valor):
                with self.assertRaises(ValueError):
                    validar_csv(self.csv(tabla), self.limites)

    def test_archivo_vacio_sin_registros_codificacion_y_tamano(self):
        casos = [
            b"", b"  ", self.csv(self.X.head(0)),
            b"\xff\xfe", b"x" * (MAX_BYTES_CSV + 1),
        ]
        for contenido in casos:
            with self.subTest(bytes=len(contenido)):
                with self.assertRaises(ValueError):
                    validar_csv(contenido, self.limites)

    def test_salida_mapea_clases_a_bar_y_probabilidades(self):
        resultado = predecir_provisional(self.modelo, self.X.head(3))
        self.assertEqual(len(resultado), 3)
        self.assertEqual(resultado["Acumulador (bar)"].tolist(), [90] * 3)
        probas = resultado[["90 bar (%)", "100 bar (%)", "115 bar (%)", "130 bar (%)"]]
        np.testing.assert_allclose(probas.sum(axis=1), 100)

    def test_baja_confianza_segun_umbral(self):
        class ModeloFijo:
            classes_ = np.array([0, 1, 2, 3])

            def predict_proba(self, X):
                return np.array([
                    [0.40, 0.30, 0.20, 0.10],
                    [config.UMBRAL_CONFIANZA, 0.20, 0.10, 0.10],
                    [0.05, 0.05, 0.05, 0.85],
                ])[:len(X)]

            def predict(self, X):
                return self.classes_[self.predict_proba(X).argmax(axis=1)]

        resultado = predecir_provisional(ModeloFijo(), self.X.head(3))
        self.assertEqual(resultado["Baja confianza"].tolist(), [True, False, False])
        # El Dummy provisional siempre tiene probabilidad 1, nunca baja confianza.
        real = predecir_provisional(self.modelo, self.X.head(3))
        self.assertFalse(real["Baja confianza"].any())

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
        self.assertEqual(resultado.iloc[0]["Acumulador (bar)"], 90)
        # La respuesta permanece al actualizar, pero desaparece al cambiar de entrada.
        app.run()
        self.assertIsNotNone(app.session_state["resultado"])
        self.assertTrue(any("Probabilidad provisional" in d.value.columns for d in app.dataframe))
        app.radio[0].set_value("Subir CSV").run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(analizar().disabled)
        self.assertEqual(len(app.dataframe), 0)
        self.assertFalse(any(b.label == "Cargar ejemplo" for b in app.button))
        self.assertEqual(len(app.get("download_button")), 0)


if __name__ == "__main__":
    unittest.main()
