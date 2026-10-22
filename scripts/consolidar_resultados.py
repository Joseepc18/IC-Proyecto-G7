"""Genera reports/resultados.csv uniendo reports/resultados_<nombre>.csv.

Cada persona escribe solo su archivo; la tabla unica se regenera con este script
y no se edita a mano. Falla si un archivo no tiene las columnas del contrato o si
se repite un nombre de modelo.
"""
from ic_g7 import config
from ic_g7.resultados import consolidar


def main() -> None:
    unida = consolidar()
    unida.to_csv(config.RUTA_RESULTADOS, index=False)
    print(f"{config.RUTA_RESULTADOS}: {len(unida)} filas")


if __name__ == "__main__":
    main()
