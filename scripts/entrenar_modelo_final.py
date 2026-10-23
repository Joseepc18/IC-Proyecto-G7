"""Guarda la MLP elegida con todo entrenamiento; no carga prueba."""
from ic_g7 import config
from ic_g7.modelo_final import preparar_modelo


def main():
    metadata = preparar_modelo()
    print(f"Modelo: {metadata['seleccion']['modelo']}")
    print(f"Entrenamiento: {metadata['ciclos_entrenamiento']} ciclos.")
    print(f"MLP: {metadata['iteraciones_mlp']} iteraciones, sin avisos de convergencia.")
    print(f"Artefactos en: {config.DIR_MODELOS}")
    print("Evaluación en prueba pendiente del 27/oct; Streamlit continúa con Dummy.")


if __name__ == "__main__":
    main()
