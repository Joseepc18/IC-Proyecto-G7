"""Ejecuta los experimentos de Dayron sin consultar el conjunto de prueba."""
from ic_g7.experimentos_dayron import ejecutar_experimentos


def main():
    tablas = ejecutar_experimentos()
    print(tablas["resultados"].drop(columns="hiperparámetros").to_string(index=False))


if __name__ == "__main__":
    main()
