"""La tabla unica se arma con un archivo por persona sin pisar filas de otros."""
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from ic_g7 import config
from ic_g7.evaluacion import validar_cruzado
from ic_g7.pipeline import crear_pipeline
from ic_g7.resultados import consolidar


def tabla(*modelos):
    return pd.DataFrame(
        [[m, "{}", "0.5000 ± 0.1000", "0.5000 ± 0.1000", "0.7000 ± 0.1000", 0.1, 0.01]
         for m in modelos],
        columns=config.COLUMNAS_RESULTADOS,
    )


class PruebasResultados(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_une_los_archivos_de_cada_persona_en_orden(self):
        tabla("MLP defecto").to_csv(self.dir / "resultados_jose.csv", index=False)
        tabla("KNN", "SVM").to_csv(self.dir / "resultados_luis.csv", index=False)
        unida = consolidar(self.dir)
        self.assertEqual(unida["modelo"].tolist(), ["MLP defecto", "KNN", "SVM"])
        self.assertEqual(list(unida.columns), config.COLUMNAS_RESULTADOS)

    def test_no_altera_los_decimales_de_los_tiempos(self):
        original = tabla("KNN").assign(t_entrenamiento=0.01621856689453125)
        original.to_csv(self.dir / "resultados_luis.csv", index=False)
        unida = consolidar(self.dir)
        self.assertEqual(unida.loc[0, "t_entrenamiento"], 0.01621856689453125)
        unida.to_csv(self.dir / "salida.csv", index=False)
        self.assertEqual(
            (self.dir / "salida.csv").read_text(), (self.dir / "resultados_luis.csv").read_text()
        )

    def test_no_relee_la_tabla_unica_ya_generada(self):
        tabla("KNN").to_csv(self.dir / "resultados_luis.csv", index=False)
        tabla("KNN", "viejo").to_csv(self.dir / "resultados.csv", index=False)
        self.assertEqual(consolidar(self.dir)["modelo"].tolist(), ["KNN"])

    def test_rechaza_nombres_repetidos_columnas_distintas_y_sin_archivos(self):
        tabla("KNN").to_csv(self.dir / "resultados_luis.csv", index=False)
        tabla("KNN").to_csv(self.dir / "resultados_dayron.csv", index=False)
        with self.assertRaises(ValueError):
            consolidar(self.dir)
        tabla("RF").drop(columns="t_inferencia").to_csv(
            self.dir / "resultados_dayron.csv", index=False
        )
        with self.assertRaises(ValueError):
            consolidar(self.dir)
        with tempfile.TemporaryDirectory() as vacio:
            with self.assertRaises(FileNotFoundError):
                consolidar(Path(vacio))

    def escribir_y_consolidar(self, tabla_modificada, nombre="resultados_luis.csv"):
        tabla_modificada.to_csv(self.dir / nombre, index=False)
        return consolidar(self.dir)

    def test_rechaza_archivos_sin_filas_o_vacios(self):
        encabezado = ",".join(config.COLUMNAS_RESULTADOS)
        (self.dir / "resultados_luis.csv").write_text(encabezado + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "resultados_luis.csv.*no tiene filas"):
            consolidar(self.dir)
        (self.dir / "resultados_luis.csv").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "resultados_luis.csv.*vacio"):
            consolidar(self.dir)

    def test_rechaza_modelo_vacio_o_en_blanco_con_archivo_y_fila(self):
        for valor in (None, "   "):
            malo = tabla("KNN", "SVM")
            malo.loc[1, "modelo"] = valor
            with self.assertRaisesRegex(ValueError, "resultados_luis.csv, fila 3.*modelo"):
                self.escribir_y_consolidar(malo)

    def test_rechaza_metricas_ausentes_o_mal_formadas(self):
        for columna in ("F1_macro_CV (media±std)", "exactitud_CV", "ROC_AUC_CV"):
            for valor in (None, "sin resultado", "0.5", "0.5 ± ", "nan ± 0.1", "-0.5 ± 0.1"):
                malo = tabla("KNN", "SVM")
                malo.loc[0, columna] = valor
                with self.assertRaisesRegex(ValueError, f"fila 2.*{re.escape(columna)}"):
                    self.escribir_y_consolidar(malo)

    def test_rechaza_tiempos_ausentes_no_numericos_no_finitos_o_negativos(self):
        for columna in ("t_entrenamiento", "t_inferencia"):
            for valor in (None, "n/d", np.inf, -np.inf, -0.5):
                malo = tabla("KNN", "SVM")
                malo["t_entrenamiento"] = malo["t_entrenamiento"].astype(object)
                malo["t_inferencia"] = malo["t_inferencia"].astype(object)
                malo.loc[1, columna] = valor
                with self.assertRaisesRegex(ValueError, f"fila 3.*{re.escape(columna)}"):
                    self.escribir_y_consolidar(malo)

    def test_acepta_tiempo_cero(self):
        self.assertEqual(
            self.escribir_y_consolidar(tabla("KNN").assign(t_inferencia=0.0)).shape[0], 1
        )

    def test_validar_cruzado_genera_las_columnas_del_contrato(self):
        # 4 clases x 10 grupos x 2 ciclos: cada pliegue ve las 4 clases.
        grupos = pd.Series(np.repeat(np.arange(40), 2))
        y = pd.Series(np.repeat(np.tile(np.arange(4), 10), 2))
        X = pd.DataFrame(np.random.default_rng(config.RANDOM_STATE).normal(size=(80, 3)))
        modelo = crear_pipeline(DummyClassifier(strategy="stratified",
                                                random_state=config.RANDOM_STATE))
        fila, por_pliegue = validar_cruzado(modelo, X, y, grupos, "Dummy")
        self.assertEqual(list(fila), config.COLUMNAS_RESULTADOS)
        self.assertEqual(len(por_pliegue), config.N_PLIEGUES)
        self.assertGreater(fila["t_inferencia"], 0)


if __name__ == "__main__":
    unittest.main()
