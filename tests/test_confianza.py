import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier

from ic_g7 import config
from ic_g7.confianza import (
    UMBRALES, elegir_umbral, probabilidades_fuera_de_pliegue, tabla_umbrales,
)


class PruebasConfianza(unittest.TestCase):
    def setUp(self):
        # 4 ciclos: dos seguros y correctos, uno dudoso correcto y uno dudoso errado.
        self.probas = np.array([
            [0.95, 0.03, 0.01, 0.01],
            [0.02, 0.90, 0.04, 0.04],
            [0.40, 0.35, 0.15, 0.10],
            [0.30, 0.45, 0.15, 0.10],
        ])
        self.y = np.array([0, 1, 0, 0])

    def test_tabla_calcula_cobertura_y_exactitud(self):
        tabla = tabla_umbrales(self.probas, self.y, umbrales=[0.25, 0.50, 0.92, 0.99])
        self.assertEqual(tabla["aceptados"].tolist(), [4, 2, 1, 0])
        np.testing.assert_allclose(tabla["cobertura"], [1.0, 0.5, 0.25, 0.0])
        np.testing.assert_allclose(tabla["exactitud"][:3], [0.75, 1.0, 1.0])
        self.assertTrue(np.isnan(tabla["exactitud"].iloc[3]))

    def test_umbral_incluido_se_acepta(self):
        tabla = tabla_umbrales(self.probas, self.y, umbrales=[0.90])
        self.assertEqual(tabla["aceptados"].iloc[0], 2)

    def test_elige_el_umbral_mas_bajo_que_cumple_el_objetivo(self):
        tabla = tabla_umbrales(self.probas, self.y)
        umbral, alcanzado, _ = elegir_umbral(tabla, objetivo=0.90)
        self.assertTrue(alcanzado)
        # Solo los dos ciclos seguros llegan al 100 %: el 0.46 ya deja fuera al dudoso errado.
        self.assertAlmostEqual(umbral, 0.46)

    def test_sin_objetivo_alcanzable_usa_la_mejor_relacion(self):
        # Ningun umbral supera 0.5 de exactitud; aceptar todo da la mejor relacion.
        probas = np.array([[0.4, 0.6], [0.4, 0.6], [0.9, 0.1], [0.9, 0.1]])
        y = np.array([1, 0, 0, 1])
        tabla = tabla_umbrales(probas, y, umbrales=[0.5, 0.7, 0.95])
        umbral, alcanzado, criterio = elegir_umbral(tabla, objetivo=0.99)
        self.assertFalse(alcanzado)
        self.assertIn("armonica", criterio)
        self.assertEqual(umbral, 0.5)

    def test_sin_aceptados_da_error(self):
        tabla = tabla_umbrales(self.probas, self.y, umbrales=[0.99])
        with self.assertRaises(ValueError):
            elegir_umbral(tabla)

    def test_entradas_invalidas_dan_error_claro(self):
        buenas = [[0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]]
        casos = {
            "una etiqueta para dos filas": (buenas, [0], "una etiqueta por ciclo"),
            "dos etiquetas para una fila": (buenas[:1], [0, 1], "una etiqueta por ciclo"),
            "matriz vacia": (np.empty((0, 4)), [], "no vacia"),
            "vector en vez de matriz": ([0.9, 0.1], [0, 1], "matriz"),
            "valor no finito": ([[np.nan, 0.5, 0.5, 0.0]] + buenas[1:], [0, 1], "no finitos"),
            "probabilidad fuera de rango": ([[1.2, -0.2, 0.0, 0.0]] + buenas[1:], [0, 1], "entre 0 y 1"),
            "no suma uno": ([[0.5, 0.1, 0.0, 0.0]] + buenas[1:], [0, 1], "sumar 1"),
            "clase inexistente": (buenas, [0, 4], "enteros entre 0 y 3"),
            "clase negativa": (buenas, [0, -1], "enteros entre 0 y 3"),
            "etiquetas decimales": (buenas, [0.0, 1.0], "enteros entre 0 y 3"),
        }
        for nombre, (probas, y, mensaje) in casos.items():
            with self.subTest(nombre):
                with self.assertRaisesRegex(ValueError, mensaje):
                    tabla_umbrales(probas, y, umbrales=[0.5])

    def test_la_cobertura_nunca_supera_uno(self):
        tabla = tabla_umbrales(self.probas, self.y)
        self.assertTrue((tabla["cobertura"] <= 1).all())

    def test_se_detiene_si_la_mlp_no_converge(self):
        rng = np.random.default_rng(config.RANDOM_STATE)
        X = rng.normal(size=(60, 5))
        y = np.arange(60) % 4
        grupos = np.arange(60) % 12
        pliegues = list(StratifiedGroupKFold(n_splits=3).split(X, y, grupos))
        # max_iter=1 obliga a MLPClassifier a avisar que no convergio.
        no_converge = MLPClassifier(max_iter=1, random_state=config.RANDOM_STATE)
        with self.assertRaisesRegex(RuntimeError, "avisos de convergencia"):
            probabilidades_fuera_de_pliegue(no_converge, X, y, grupos, pliegues)

    def test_devuelve_probabilidades_validas_si_converge(self):
        X = np.vstack([np.full((20, 2), c) for c in range(4)]) + np.random.default_rng(
            config.RANDOM_STATE).normal(scale=0.05, size=(80, 2))
        y = np.repeat(np.arange(4), 20)
        grupos = np.arange(80) % 16
        pliegues = list(StratifiedGroupKFold(n_splits=2).split(X, y, grupos))
        modelo = LogisticRegression(max_iter=1000)
        probabilidades = probabilidades_fuera_de_pliegue(modelo, X, y, grupos, pliegues)
        self.assertEqual(probabilidades.shape, (80, 4))
        np.testing.assert_allclose(probabilidades.sum(axis=1), 1)

    def test_rejilla_cubre_el_rango_util(self):
        self.assertAlmostEqual(UMBRALES[0], 0.25)
        self.assertAlmostEqual(UMBRALES[-1], 0.99)

    def test_umbral_configurado_es_una_probabilidad_valida(self):
        self.assertGreater(config.UMBRAL_CONFIANZA, 0.25)
        self.assertLess(config.UMBRAL_CONFIANZA, 1.0)


if __name__ == "__main__":
    unittest.main()
