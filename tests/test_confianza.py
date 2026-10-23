import unittest

import numpy as np

from ic_g7 import config
from ic_g7.confianza import UMBRALES, elegir_umbral, tabla_umbrales


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

    def test_rejilla_cubre_el_rango_util(self):
        self.assertAlmostEqual(UMBRALES[0], 0.25)
        self.assertAlmostEqual(UMBRALES[-1], 0.99)

    def test_umbral_configurado_es_una_probabilidad_valida(self):
        self.assertGreater(config.UMBRAL_CONFIANZA, 0.25)
        self.assertLess(config.UMBRAL_CONFIANZA, 1.0)


if __name__ == "__main__":
    unittest.main()
