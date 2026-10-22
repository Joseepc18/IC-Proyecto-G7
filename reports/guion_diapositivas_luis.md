# Guion de presentación G7 (borrador de Luis, 22/oct/2026)

**Límite:** 10 diapositivas y 10 minutos de exposición; después, 5 minutos de
preguntas. Todos los integrantes deben participar. Este guion tiene **9
diapositivas** y no sustituye la presentación final: faltan RF, MLP, GA,
selección conjunta, prueba única y capturas del prototipo final. No insertar
resultados de prueba antes de la reunión acordada para el 27/oct.

| N.º | Título y mensaje central | Evidencia visual | Estado |
|---:|---|---|---|
| 1 | **Clasificación de presión del acumulador hidráulico.** Problema y autores. | Imagen sencilla del circuito o recurso SVG de la app. | Listo para maquetar. |
| 2 | **Cuatro niveles con distinta gravedad.** La predicción apoya al técnico; 90 bar es la condición más crítica. | Escala 90 → 100 → 115 → 130 bar. | Listo. |
| 3 | **Datos y prevención de fugas.** 2.205 ciclos, 17 sensores, 68 variables; 1.827 train / 378 prueba; separación por combinaciones completas. | Diagrama de flujo: datos → estadísticas por ciclo → split por grupos → Pipeline. | Justificación temporal pendiente de acuerdo del equipo. |
| 4 | **EDA: grupos grandes y clases desiguales.** Tres grupos suman 36,8 % de entrenamiento y dominan tres pliegues. | Gráfico de clases y tamaños de grupo del `01_eda.ipynb`. | Listo. |
| 5 | **Método experimental común.** Escalado y filtro dentro del Pipeline; cinco pliegues por grupos; F1 macro principal; Dummy, KNN, SVM, RF, MLP y GA. | Esquema simple del Pipeline y matriz de modelos. | Completar detalles GA con José. |
| 6 | **Comparación de modelos.** La tabla única permite elegir por F1, dispersión, tiempo y errores graves. | Tabla de `reports/resultados.csv` con todos los modelos. | Faltan RF, MLP y GA de Dayron/José. |
| 7 | **Qué errores cometen los candidatos.** Evitar que 90 bar se informe como 130 bar. | Matrices de confusión y recall por clase; hoy existe la figura KNN/SVM. | Actualizar con el modelo final. |
| 8 | **Modelo final y prototipo.** Una predicción, probabilidades, baja confianza y rechazo de CSV inválidos. | Capturas de tres casos, incluido uno límite. | Depende de selección, umbral y app final. |
| 9 | **Conclusiones y límites.** Resultado en prueba separado de CV, alcance de un solo banco, revisión humana y mejoras futuras. | Tres conclusiones breves y fuente UCI. | Completar tras la prueba única. |

## Cifras verificadas que pueden usarse ya

- Entrenamiento: 1.827 ciclos, 68 variables, 111 grupos; cuatro clases con
  conteos 686 / 317 / 317 / 507 en orden 90 / 100 / 115 / 130 bar.
- Tres grupos mayores: 672 ciclos, 36,8 % del entrenamiento.
- F1 macro por CV: Dummy estratificado 0,250 ± 0,028; KNN ajustado 0,528 ±
  0,107; SVM RBF ajustada 0,767 ± 0,087.
- En las predicciones fuera de pliegue de SVM, 69 de 686 ciclos de 90 bar se
  etiquetan como 130 bar. Este error debe discutirse aunque la media de F1 sea
  favorable.

## Información necesaria para finalizar

1. **Dayron:** filas de RF y MLP con los cinco pliegues, métricas por clase,
   matrices y tiempos; capturas del prototipo final.
2. **José:** MLP optimizada por GA y búsqueda aleatoria, tres corridas de cada
   una, curva de convergencia y criterio de baja confianza.
3. **Equipo completo:** elección del único modelo final y evaluación única en
   prueba, además de la justificación acordada de la partición por grupos ante
   la indicación de partición temporal del PDF.

**Fuente de los datos:** Helwig, Pignanelli y Schütze, *Condition Monitoring of
Hydraulic Systems*, UCI Machine Learning Repository,
https://doi.org/10.24432/C5CW21. Declarar en el anexo final el uso de Codex
en programación, análisis y redacción de este borrador.
