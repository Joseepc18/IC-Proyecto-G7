"""Dibuja la matriz ya guardada; no carga ni vuelve a evaluar el modelo."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ic_g7 import config


def main():
    carpeta = config.DIR_REPORTES / "evaluacion_final"
    informe = json.loads((carpeta / "informe.json").read_text(encoding="utf-8"))
    matriz = np.asarray(informe["matriz_confusion"], dtype=int)
    etiquetas = [str(bar) for bar in informe["orden_matriz_bar"]]
    if matriz.shape != (len(etiquetas), len(etiquetas)):
        raise ValueError("La matriz no coincide con el orden de categorías guardado.")

    fig, ax = plt.subplots(figsize=(7.4, 6.4), layout="constrained")
    imagen = ax.imshow(matriz, cmap="Blues", vmin=0)
    fig.colorbar(imagen, ax=ax, label="Número de ciclos", shrink=0.85)
    ax.set(
        xticks=range(len(etiquetas)), xticklabels=etiquetas,
        yticks=range(len(etiquetas)), yticklabels=etiquetas,
        xlabel="Clase predicha (bar)", ylabel="Clase real (bar)",
        title=(
            f'{informe["modelo"]}: evaluación final\n'
            f'{informe["ciclos_prueba"]} ciclos · {informe["grupos_prueba"]} grupos'
        ),
    )
    for fila, columna in np.ndindex(matriz.shape):
        ax.text(
            columna, fila, str(matriz[fila, columna]), ha="center", va="center",
            color="white" if matriz[fila, columna] > matriz.max() / 2 else "#14213d",
            fontsize=13,
        )
    fig.suptitle(
        f'F1 macro: {informe["f1_macro_prueba"]:.4f}  |  '
        f'Exactitud: {informe["exactitud_prueba"]:.2%}', fontsize=11,
    )
    destino = carpeta / "matriz_confusion.png"
    fig.savefig(destino, dpi=200)
    plt.close(fig)
    print(f"Matriz guardada en {destino}")


if __name__ == "__main__":
    main()
