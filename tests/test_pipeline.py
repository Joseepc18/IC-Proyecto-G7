"""El filtro se ajusta con las filas recibidas y conserva el contrato del Pipeline."""

import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from ic_g7.pipeline import FiltroCorrelacion, crear_pipeline


class PruebasPipeline(unittest.TestCase):
    def test_elimina_redundancia_y_constantes_sin_perder_nombres(self):
        X = pd.DataFrame({
            "presion": [0, 1, 2, 3, 4, 5, 6, 7],
            "copia": [0, 1, 2, 3, 4, 5, 6, 7],
            "otra": [0, 1, 0, 1, 1, 0, 1, 0],
            "constante": [2] * 8,
        })
        y = [0, 0, 0, 0, 1, 1, 1, 1]
        modelo = crear_pipeline(LogisticRegression(), umbral_correlacion=0.95)
        modelo.fit(X, y)

        self.assertEqual(
            modelo[:-1].get_feature_names_out().tolist(), ["presion", "otra"]
        )
        self.assertEqual(modelo.predict_proba(X).shape, (8, 2))

    def test_cada_ajuste_recalcula_el_filtro(self):
        base = pd.DataFrame({
            "a": np.arange(8),
            "b": np.arange(8),
            "c": [0, 1, 0, 1, 1, 0, 1, 0],
        })
        cambiado = base.copy()
        cambiado["b"] = [3, 0, 7, 2, 5, 1, 6, 4]
        y = [0, 0, 0, 0, 1, 1, 1, 1]
        plantilla = crear_pipeline(LogisticRegression(), umbral_correlacion=0.95)

        primer_ajuste = clone(plantilla).fit(base, y)
        segundo_ajuste = clone(plantilla).fit(cambiado, y)
        self.assertEqual(primer_ajuste[:-1].get_feature_names_out().tolist(), ["a", "c"])
        self.assertEqual(
            segundo_ajuste[:-1].get_feature_names_out().tolist(), ["a", "b", "c"]
        )

    def test_transform_devuelve_solo_las_columnas_conservadas(self):
        X = pd.DataFrame({
            "presion": [0, 1, 2, 3, 4, 5, 6, 7],
            "copia": [0, 1, 2, 3, 4, 5, 6, 7],
            "otra": [0, 1, 0, 1, 1, 0, 1, 0],
            "constante": [2] * 8,
        })
        filtro = FiltroCorrelacion(umbral=0.95).fit(X)
        self.assertEqual(filtro.transform(X).shape, (8, 2))
        self.assertEqual(filtro.get_support().tolist(), [True, False, True, False])

    def test_constante_en_primera_columna_se_elimina(self):
        X = pd.DataFrame({
            "constante": [2] * 8,
            "presion": [0, 1, 2, 3, 4, 5, 6, 7],
            "otra": [0, 1, 0, 1, 1, 0, 1, 0],
        })
        filtro = FiltroCorrelacion(umbral=0.95).fit(X)
        self.assertEqual(filtro.get_feature_names_out().tolist(), ["presion", "otra"])

    def test_transform_usa_la_mascara_aprendida_en_fit(self):
        base = pd.DataFrame({
            "a": np.arange(8),
            "b": np.arange(8),
            "c": [0, 1, 0, 1, 1, 0, 1, 0],
        })
        otros = base.assign(b=[3, 0, 7, 2, 5, 1, 6, 4])
        filtro = FiltroCorrelacion(umbral=0.95).fit(base)
        self.assertEqual(filtro.transform(otros).shape, (8, 2))

    def test_umbral_invalido_y_ancho_distinto_se_rechazan(self):
        X = pd.DataFrame({"a": np.arange(8), "b": [0, 1, 0, 1, 1, 0, 1, 0]})
        for umbral in (0, -0.5, 1.5):
            with self.subTest(umbral=umbral):
                with self.assertRaises(ValueError):
                    FiltroCorrelacion(umbral=umbral).fit(X)
        filtro = FiltroCorrelacion(umbral=0.95).fit(X)
        with self.assertRaises(ValueError):
            filtro.transform(X[["a"]])


if __name__ == "__main__":
    unittest.main()
