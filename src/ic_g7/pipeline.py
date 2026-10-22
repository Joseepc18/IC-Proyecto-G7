"""Preprocesamiento compartido para los clasificadores del proyecto.

El escalado y la selección se ajustan dentro de cada pliegue de entrenamiento
cuando el Pipeline se usa con la validación cruzada de ``evaluacion.py``.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted


class FiltroCorrelacion(TransformerMixin, BaseEstimator):
    """Conserva variables que no duplican a otra ya seleccionada.

El orden de las variables determina cuál se conserva. El filtro no consulta
las etiquetas y aprende las correlaciones únicamente en ``fit``.
    """

    def __init__(self, umbral: float = 0.995):
        self.umbral = umbral

    def fit(self, X, y=None):
        if not 0 < self.umbral <= 1:
            raise ValueError("El umbral de correlación debe estar en (0, 1].")

        valores = check_array(X, dtype=np.float64, ensure_all_finite=True)
        self.n_features_in_ = valores.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        # Las columnas constantes no aportan información y no tienen una
        # correlación de Pearson definida.
        variables = np.flatnonzero(np.std(valores, axis=0) > 0)
        if not len(variables):
            raise ValueError("Todas las variables son constantes en este entrenamiento.")

        seleccionadas = []
        if len(variables) == 1:
            seleccionadas = variables.tolist()
        else:
            correlaciones = np.abs(np.corrcoef(valores[:, variables], rowvar=False))
            for posicion, indice in enumerate(variables):
                if all(correlaciones[posicion, anterior] <= self.umbral
                       for anterior in seleccionadas):
                    seleccionadas.append(posicion)
            seleccionadas = variables[seleccionadas].tolist()

        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[seleccionadas] = True
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        valores = check_array(X, dtype=np.float64, ensure_all_finite=True)
        if valores.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Se esperaban {self.n_features_in_} variables; llegaron {valores.shape[1]}."
            )
        return valores[:, self.support_]

    def get_support(self):
        """Máscara de las variables conservadas tras ``fit``."""
        check_is_fitted(self, "support_")
        return self.support_.copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "support_")
        if input_features is None:
            input_features = getattr(
                self, "feature_names_in_", [f"x{i}" for i in range(self.n_features_in_)]
            )
        nombres = np.asarray(input_features, dtype=object)
        if len(nombres) != self.n_features_in_:
            raise ValueError("El número de nombres no coincide con las variables de entrada.")
        return nombres[self.support_]


def crear_pipeline(estimador, umbral_correlacion: float = 0.995) -> Pipeline:
    """Crea escalador, filtro de redundancia y estimador en un solo Pipeline.

    El umbral se aplica a ``|r|``: se elimina una variable si supera el umbral
    con otra ya conservada. ``1.0`` permite comparar sin eliminar variables
    correlacionadas, aunque se siguen retirando las constantes.
    """
    return Pipeline([
        ("escalador", StandardScaler()),
        ("correlacion", FiltroCorrelacion(umbral=umbral_correlacion)),
        ("modelo", estimador),
    ])
