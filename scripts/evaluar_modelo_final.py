"""Evaluación final única del modelo elegido sobre la partición reservada."""
import argparse
from ic_g7.modelo_final import evaluar_prueba_una_vez


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirmar-evaluacion", action="store_true", required=True,
        help="Confirma la evaluación única acordada y que el modelo ya fue elegido.",
    )
    args = parser.parse_args()
    informe = evaluar_prueba_una_vez(confirmar_evaluacion=args.confirmar_evaluacion)
    print(f"F1 macro en prueba: {informe['f1_macro_prueba']:.4f}")
    print(f"Exactitud en prueba: {informe['exactitud_prueba']:.4f}")
    print(f"ROC-AUC en prueba: {informe['roc_auc_ovr_macro_prueba']:.4f}")
    print("Evaluación registrada. Se consulta el informe guardado; no se vuelve a ejecutar.")


if __name__ == "__main__":
    main()
