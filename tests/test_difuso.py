import unittest

import pandas as pd

from ic_g7 import config
from ic_g7.difuso import (
    VARIABLES, activacion_reglas, ajustar_referencia, explicar, pertenencia,
)

# Referencia fija (p25, p50, p75) para no depender de los datos.
REFERENCIA = {v: (10.0, 20.0, 30.0) for v in VARIABLES}


def fila(**valores):
    base = {columna: 0.0 for columna in config.COLUMNAS_ENTRADA}
    base.update({v: 20.0 for v in VARIABLES})
    base.update(valores)
    return pd.Series(base)


class PruebasDifuso(unittest.TestCase):
    def test_pertenencia_en_los_puntos_clave(self):
        ref = (10.0, 20.0, 30.0)
        self.assertEqual(pertenencia(5.0, ref), {"baja": 1.0, "media": 0.0, "alta": 0.0})
        self.assertEqual(pertenencia(20.0, ref), {"baja": 0.0, "media": 1.0, "alta": 0.0})
        self.assertEqual(pertenencia(35.0, ref), {"baja": 0.0, "media": 0.0, "alta": 1.0})
        medio = pertenencia(15.0, ref)
        self.assertAlmostEqual(medio["baja"], 0.5)
        self.assertAlmostEqual(medio["media"], 0.5)

    def test_los_grados_suman_uno_y_no_son_negativos(self):
        for valor in (-5, 10, 12.5, 20, 27, 30, 99):
            grados = pertenencia(valor, (10.0, 20.0, 30.0))
            self.assertAlmostEqual(sum(grados.values()), 1.0)
            self.assertTrue(all(g >= 0 for g in grados.values()))

    def test_percentiles_iguales_no_dividen_por_cero(self):
        grados = pertenencia(5.0, (7.0, 7.0, 7.0))
        self.assertAlmostEqual(sum(grados.values()), 1.0)

    def test_referencia_usa_percentiles_del_dataframe_recibido(self):
        X = pd.DataFrame({v: range(101) for v in VARIABLES})
        self.assertEqual(ajustar_referencia(X)["PS5_max"], (25.0, 50.0, 75.0))

    def test_tres_presiones_bajas_apuntan_a_optima(self):
        f = fila(PS5_max=0.0, PS3_mean=5.0, PS1_mean=9.0)
        self.assertEqual(activacion_reglas(f, REFERENCIA)["optima"], 1.0)
        frase = explicar(f, REFERENCIA)
        self.assertIn("bajas", frase)
        self.assertIn("130 bar", frase)

    def test_una_presion_alta_apunta_a_menos_de_130(self):
        f = fila(PS5_max=40.0)
        frase = explicar(f, REFERENCIA)
        self.assertIn("PS5_max está alta", frase)
        self.assertIn("por debajo de 130 bar", frase)
        self.assertNotIn("PS3_mean", frase)

    def test_varias_presiones_altas_se_nombran_todas(self):
        frase = explicar(fila(PS5_max=40.0, PS1_mean=33.0), REFERENCIA)
        self.assertIn("PS5_max y PS1_mean están altas", frase)

    def test_valores_intermedios_no_indican_nivel(self):
        frase = explicar(fila(), REFERENCIA)
        self.assertIn("no indican un nivel de presión claro", frase)

    def test_avisa_si_la_pista_contradice_la_prediccion(self):
        optima = fila(PS5_max=0.0, PS3_mean=5.0, PS1_mean=9.0)
        self.assertIn("no coincide", explicar(optima, REFERENCIA, clase_predicha=0))
        self.assertNotIn("no coincide", explicar(optima, REFERENCIA, clase_predicha=3))
        alta = fila(PS5_max=40.0)
        self.assertIn("no coincide", explicar(alta, REFERENCIA, clase_predicha=3))
        self.assertNotIn("no coincide", explicar(alta, REFERENCIA, clase_predicha=1))

    def test_acepta_diccionario_ademas_de_serie(self):
        self.assertIsInstance(explicar(fila().to_dict(), REFERENCIA), str)


if __name__ == "__main__":
    unittest.main()
