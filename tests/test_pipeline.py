"""El filtro se ajusta con las filas recibidas y conserva el contrato del Pipeline."""

import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from ic_g7.pipeline import crear_pipeline


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


if __name__ == "__main__":
    unittest.main()
