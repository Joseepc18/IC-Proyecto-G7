# Guion de presentación G7 (actualizado, 23/oct/2026)

**Límite:** 10 diapositivas y 10 minutos de exposición; después, 5 minutos de
preguntas. Todos los integrantes deben participar. Este guion tiene **9
diapositivas**. Ya incorpora los resultados de CV y el modelo seleccionado;
faltan la evaluación única en prueba que hará Dayron el 27/oct y las capturas
del prototipo final. Ninguna cifra de prueba debe completarse por anticipado.

| N.º | Título y mensaje central | Evidencia visual | Estado |
|---:|---|---|---|
| 1 | **Clasificación de presión del acumulador hidráulico.** Problema y autores. | Imagen sencilla del circuito o recurso SVG de la app. | Listo para maquetar. |
| 2 | **Cuatro niveles con distinta gravedad.** La predicción apoya al técnico; 90 bar es la condición más crítica. | Escala 90 → 100 → 115 → 130 bar. | Listo. |
| 3 | **Datos y prevención de fugas.** 2.205 ciclos, 17 sensores, 68 variables; 1.827 train / 378 prueba. Cada ciclo de 60 s se resume en una fila; la partición por combinaciones evita que una misma configuración aparezca a ambos lados. No se usan orden temporal ni rezagos. | Diagrama de flujo: datos → estadísticas por ciclo → split por grupos → Pipeline. | Justificación de la partición acordada por el equipo. |
| 4 | **EDA: grupos grandes y clases desiguales.** Tres grupos suman 36,8 % de entrenamiento y dominan tres pliegues. | Gráfico de clases y tamaños de grupo del `01_eda.ipynb`. | Listo. |
| 5 | **Método experimental común.** Escalado y filtro dentro del Pipeline; cinco pliegues por grupos; F1 macro principal; Dummy, KNN, SVM, RF y MLP. GA y búsqueda aleatoria: 100 evaluaciones por corrida, semillas 42–44. | Pipeline y esquema del presupuesto igual de búsqueda. | Listo para maquetar. |
| 6 | **Comparación y selección.** MLP+GA (semilla 42) alcanza 0,812 ± 0,054 de F1 macro CV, el valor más alto de la tabla. La MLP por defecto de Dayron y la de José con semilla 42 son la misma configuración. | Tabla compacta de `reports/resultados.csv`: Dummy, KNN, SVM, RF ajustado, MLP base, búsqueda aleatoria 42 y GA 42. | Datos de CV listos. Omitir comparación directa de tiempos por distinto paralelismo. |
| 7 | **Errores de la opción elegida.** MLP+GA 42 detecta 551 de 686 ciclos de 90 bar y confunde 2 con 130 bar. | Matriz de confusión de `reports/matrices_confusion_jose.csv` y recall por clase. | Datos de CV listos; añadir prueba cuando Dayron la entregue. |
| 8 | **Confianza y explicación.** Umbral 0,87: acepta 66,0 % de ciclos con 90,1 % de exactitud OOF; la regla difusa con PS5_max, PS3_mean y PS1_mean solo orienta 130 bar frente a menos de 130. | Curva de confianza de José y captura de la app con predicción, probabilidades, pista y caso inválido. | Análisis listo; faltan capturas de la app final. |
| 9 | **Conclusiones y límites.** GA supera por 0,0032 a búsqueda aleatoria en promedio de tres corridas: no se observó una mejora clara. Separar CV de la prueba única; reconocer un solo banco y necesidad de revisión humana. | Tres conclusiones breves, resultado de prueba y fuente UCI. | Completar tras la evaluación única del 27/oct. |

## Cifras verificadas que pueden usarse ya

- Entrenamiento: 1.827 ciclos, 68 variables, 111 grupos; cuatro clases con
  conteos 686 / 317 / 317 / 507 en orden 90 / 100 / 115 / 130 bar.
- Tres grupos mayores: 672 ciclos, 36,8 % del entrenamiento.
- F1 macro por CV: Dummy estratificado 0,250 ± 0,028; KNN ajustado 0,528 ±
  0,107; SVM RBF 0,767 ± 0,087; RF ajustado 0,715 ± 0,140; MLP de referencia
  0,725 ± 0,089; MLP + búsqueda aleatoria 42 0,808 ± 0,077; MLP+GA 42
  0,812 ± 0,054. La tabla completa tiene 16 filas, incluidas tres semillas.
- Entre semillas, F1 macro medio: GA 0,8093 frente a búsqueda aleatoria
  0,8061; diferencia 0,0032, menor que la dispersión entre pliegues.
- En predicciones fuera de pliegue, MLP+GA 42 etiquetó 2 de los 686 ciclos
  reales de 90 bar como 130 bar; su recall de 90 bar es 0,803.
- Umbral 0,87: 1.206 de 1.827 ciclos aceptados (66,0 %) y exactitud 90,1 %
  entre ellos, con un intervalo por remuestreo de grupos de 84,5 a 94,6 %.
  Son estimaciones de entrenamiento, no resultados de prueba.

## Información necesaria para finalizar

1. **Dayron:** evaluación única en prueba del modelo MLP+GA 42 el 27/oct,
   métricas por clase y matriz separadas de CV; capturas de al menos tres
   pruebas del prototipo final.
2. **Luis y equipo:** incorporar esas cifras al informe y a las diapositivas,
   comprobar el criterio (b), revisar el tiempo de exposición y ensayar los
   tres integrantes antes de entregar el 29/oct.

**Fuente de los datos:** Helwig, Pignanelli y Schütze, *Condition Monitoring of
Hydraulic Systems*, UCI Machine Learning Repository,
https://doi.org/10.24432/C5CW21. Declarar en el anexo final el uso de Codex
en programación, análisis y redacción de este guion (22 y 23/oct/2026).
