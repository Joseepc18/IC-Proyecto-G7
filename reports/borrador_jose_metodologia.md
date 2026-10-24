# Aporte de José al informe técnico: Metodología de optimización (borrador, 23/oct/2026)

Este texto cubre la metodología de optimización de hiperparámetros, la selección del modelo final, el umbral de baja confianza y la explicación difusa. Las cifras proceden de `reports/resultados.csv`, `reports/resultados_jose.csv`, `reports/pliegues_jose.csv` y de los notebooks `03_metaheuristica`, `04_umbral_confianza` y `05_explicacion_difusa`. Todo se calculó con los 1.827 ciclos de entrenamiento: la prueba reservada no se cargó ni se consultó.

## Metodología (optimización)

### 1. Objetivo y qué se optimiza

Se optimizan los hiperparámetros de una red neuronal multicapa (`MLPClassifier` de scikit-learn; Pedregosa et al., 2011) que clasifica la presión del acumulador (90, 100, 115 y 130 bar) a partir de las 68 variables de cada ciclo. Se eligió la MLP porque mezcla hiperparámetros enteros, continuos y categóricos, donde una búsqueda evolutiva tiene sentido. La pregunta es si un algoritmo genético (GA) mejora a la búsqueda convencional con el mismo presupuesto.

| Gen | Tipo | Rango |
|---|---|---|
| `hidden_layer_sizes` | categórico | (32,), (64,), (128,), (64, 32), (128, 64) |
| `alpha` | continuo, escala logarítmica | [1e-5, 1e-1] |
| `learning_rate_init` | continuo, escala logarítmica | [1e-4, 1e-2] |
| `activation` | categórico | relu, tanh |

`max_iter` queda fijo en 4.000 para todos los métodos. En un sondeo de las 40 esquinas del espacio, con 1.000 iteraciones las 20 pruebas con `learning_rate_init = 1e-4` emitieron `ConvergenceWarning`; con 4.000 no hubo avisos en 120 pruebas (máximo usado: 2.859). Con un valor corto, el GA premiaría configuraciones que solo entrenan más rápido.

### 2. Algoritmo genético con PyGAD

Cada individuo tiene cuatro genes: un índice entero para la arquitectura (0 a 4); los exponentes decimales de `alpha` (−5 a −1) y de `learning_rate_init` (−4 a −2), transformados con 10^x para obtener la escala logarítmica; y un índice (0 o 1) para la activación. La aptitud es el F1 macro medio de la validación cruzada por grupos de cinco pliegues. El `random_state` de la MLP se fija igual a la semilla de la corrida, de modo que un individuo siempre recibe la misma aptitud y el GA no persigue ruido.

Se usa PyGAD 3.9.2 (Gad, 2024) con población de 10, 9 generaciones, 4 padres por generación, selección `sss`, cruce de un punto y mutación aleatoria de un gen por descendiente. Sin élite ni padres conservados (`keep_elitism=0`, `keep_parents=0`), cada generación evalúa 10 individuos nuevos: 10 + 9 × 10 = 100 llamadas a la aptitud, contadas dentro de la propia función. Como sin élite el mejor individuo final puede ser peor que uno anterior, se toma la mejor configuración de las 100 evaluadas. Los operadores no se ajustaron. Entorno: Python 3.12.10, scikit-learn 1.9.1, PyGAD 3.9.2, numpy 2.5.4, pandas 3.0.6.

### 3. Línea de comparación

`RandomizedSearchCV` explora el mismo espacio con 100 iteraciones (`loguniform` en `alpha` y `learning_rate_init`, elección uniforme en las categorías), el mismo `Pipeline`, los mismos pliegues y `refit=False` (Bergstra y Bengio, 2012). Se añade la MLP por defecto de scikit-learn (100 neuronas, `alpha` 1e-4, tasa 1e-3, relu), sin búsqueda y solo con `max_iter=4000` y la semilla cambiados. Cada método se ejecutó tres veces, con las semillas 42, 43 y 44. Cada corrida de los dos métodos de búsqueda usó exactamente 100 evaluaciones de CV, tardó entre 28 y 44 minutos (tres semillas en paralelo) y no emitió avisos de convergencia. La MLP por defecto se evaluó sin búsqueda (0 evaluaciones de búsqueda y unos 0,2 minutos por corrida). Las evaluaciones adicionales para reportar métricas no forman parte del presupuesto de búsqueda.

| Método | F1 macro CV (media de 3 corridas) | Desv. entre corridas | Rango | Desv. entre pliegues (media de 3) |
|---|---|---|---|---|
| MLP por defecto | 0,7254 | 0,0049 | 0,7207 a 0,7305 | 0,0895 |
| MLP + `RandomizedSearchCV` | 0,8061 | 0,0021 | 0,8038 a 0,8080 | 0,0755 |
| MLP + GA | 0,8093 | 0,0047 | 0,8039 a 0,8123 | 0,0619 |

### 4. Validación y control de fugas

Todas las evaluaciones usan `StratifiedGroupKFold` de cinco pliegues, con barajado y semilla 42 (`RANDOM_STATE` en `config.py`), estratificado por clase y agrupado por la combinación de estados de los cuatro componentes (111 grupos en entrenamiento). Dentro de cada partición de la validación cruzada, ningún grupo está a la vez en el ajuste y en la validación, así que un ciclo nunca se valida con un modelo que vio ciclos casi idénticos; un mismo grupo sí participa en el ajuste de las otras particiones. La métrica principal es el F1 macro (media y desviación poblacional entre pliegues, más los valores por pliegue); exactitud y ROC-AUC son secundarias.

Para evitar fugas: las variables se extraen ciclo a ciclo, sin etiquetas; los estados de los componentes, `stable`, `cycle_id` y `group_id` nunca son entradas; el `StandardScaler` y el filtro de correlación (|r| > 0,995) forman parte del `Pipeline` compartido y se ajustan en cada pliegue de entrenamiento; y la partición se guarda una sola vez en `split.csv`, con un pliegue de prueba elegido solo por su estructura antes de entrenar ningún modelo final. La prueba reservada no intervino en la búsqueda, la selección ni la calibración del umbral; se usará una sola vez con el modelo ya elegido.

### 5. Selección del modelo final

Se elige por F1 macro de validación cruzada; en caso de empate, por menor desviación entre pliegues y luego por menor complejidad y tiempo de inferencia. De las 16 filas de `resultados.csv`, la de mayor F1 macro es **MLP + GA con semilla 42: 0,8123 ± 0,0543**, también la de menor desviación entre las que superan 0,80. Sus hiperparámetros son una capa de 64 neuronas, `alpha` = 0,00351, `learning_rate_init` = 0,00267, activación `tanh` y `max_iter` = 4.000. Por pliegue, su F1 macro va de 0,7303 (pliegue 4) a 0,8853 (pliegue 2). Supera el umbral del criterio (a), ≈ 0,31 (`Dummy(stratified)`: 0,2498 ± 0,0280); el criterio (b) se comprobará con la prueba reservada.

Para el criterio (c), el GA promedió 0,8093 y la búsqueda aleatoria 0,8061: una diferencia de 0,0032, muy inferior a la desviación entre pliegues (0,0755, la mayor de las dos). **No se observó una mejora clara** del GA. Las corridas de ambos métodos se solapan y con tres corridas no se afirma equivalencia estadística. Ambos superan a la MLP por defecto en unos 0,08 de F1 macro. La segunda mejor fila (GA, semilla 44: 0,8118 ± 0,0672) queda a solo 0,0005: la regla elige la semilla 42, pero esa diferencia no indica una ventaja real.

### 6. Umbral de baja confianza

La MLP final marca un ciclo como de baja confianza si su probabilidad máxima es menor que un umbral. Se calibró con probabilidades fuera de pliegue (`cross_val_predict`) de los 1.827 ciclos de entrenamiento y los mismos cinco pliegues por grupos: cada ciclo recibe su probabilidad de un modelo que no vio su combinación. Se recorrió la rejilla de 0,25 a 0,99 en pasos de 0,01 y se tomó el **menor umbral con exactitud ≥ 90 % entre los aceptados**.

| Umbral | Aceptados (de 1.827) | Cobertura | Exactitud de aceptados |
|---|---|---|---|
| sin umbral (0,25) | 1.827 | 100,0 % | 82,9 % |
| 0,60 (valor inicial) | 1.695 | 92,8 % | 85,0 % |
| 0,86 | 1.229 | 67,3 % | 90,0 % (89,99 %) |
| **0,87 (elegido)** | **1.206** | **66,0 %** | **90,1 %** |
| 0,90 | 1.124 | 61,5 % | 90,6 % |

Límites: un remuestreo de 2.000 muestras de grupos completos (semilla 42) da un intervalo del 95 % de 84,5 a 94,6 % para la exactitud de los aceptados, y solo el 54,8 % de los remuestreos llega al 90 %. Por pliegue varía de 78,9 a 97,9 %, y la clase de 115 bar es la menos aceptada (51,4 %). Las probabilidades de la MLP no están calibradas: 0,87 es un punto de corte práctico, no un «87 % de acierto». El umbral se eligió y se mide con los mismos datos, así que es algo optimista; solo la prueba reservada dirá si se mantiene.

### 7. Explicación difusa sencilla

Es una descripción de tendencias, no un clasificador. Usa tres variables de presión del análisis exploratorio: `PS5_max`, `PS3_mean` y `PS1_mean` (|r| ≤ 0,76 entre ellas). Para cada una, «baja», «media» y «alta» se definen con los percentiles 25, 50 y 75 del entrenamiento: «baja» vale 1 hasta el percentil 25 y baja a 0 en la mediana; «alta» vale 0 hasta la mediana y sube a 1 en el percentil 75; «media» es el resto.

| Variable | p25 | p50 | p75 |
|---|---|---|---|
| `PS5_max` | 8,666 | 9,168 | 9,892 |
| `PS3_mean` | 1,733 | 1,782 | 1,931 |
| `PS1_mean` | 158,110 | 158,811 | 160,914 |

Hay dos reglas, activas con grado ≥ 0,5: «óptima» (las tres variables bajas, con el mínimo de los grados) y «menor» (alguna alta, con el máximo). En entrenamiento, la regla óptima se activa en 377 ciclos, el 73,7 % de ellos de 130 bar (base: 27,8 %); la menor, en 766, con un 9,7 % de 130 bar; en 684 no se activa ninguna.

Sus límites: solo orienta «130 bar» frente a «menos de 130 bar», porque las tres variables solapan casi por completo 90, 100 y 115 bar. No explica cómo decide la MLP: el pipeline recibe las 68 variables originales y la red trabaja con las que conserva el filtro de correlación, que se aprende en cada pliegue (en el modelo final guardado conserva 41), y la red no se inspecciona; la contradice en 100 de los 377 ciclos de la regla óptima (26,5 %) y en 33 de los 766 de la menor (4,3 %), y en esos casos la MLP acierta el 74,0 % y el 63,6 %: son avisos para revisar, no veredictos. Variables y reglas se eligieron mirando el entrenamiento etiquetado, y la asociación en un solo banco no prueba causalidad.

### 8. Limitaciones y amenazas a la validez

La búsqueda y la evaluación comparten pliegues y el modelo final es el mejor de las nueve corridas de MLP, por lo que sus cifras de validación cruzada pueden ser optimistas; la única estimación independiente es la prueba reservada. Hay solo tres corridas por método, con 100 evaluaciones, población de 10 y operadores sin ajustar: la conclusión sobre el GA vale para este presupuesto y este espacio, sin prueba estadística formal, y el criterio (c) compara con la desviación entre pliegues, que no mide la variación entre corridas. Tres combinaciones suman 672 ciclos (el 30 % del total y el 36,8 % del entrenamiento) y cada una domina un pliegue, lo que explica desviaciones entre pliegues de 0,05 a 0,09 y obliga a leer los resultados por pliegue. Los datos vienen de un único banco de laboratorio con 144 combinaciones; no se probó otra instalación.

El sondeo de `max_iter` cubre las esquinas del espacio, no todos sus puntos; la ausencia de avisos se comprobó en las evaluaciones realmente ejecutadas. La MLP «por defecto» usa `max_iter=4000` y no el 200 estándar, que no converge. Los tiempos de entrenamiento se midieron con tres procesos en paralelo y no son comparables con los de otros modelos. Las trazas de las 100 evaluaciones no se guardaron en archivos y no se pueden auditar sin reejecutar (unos 90 minutos). Equipo de ejecución: procesador AMD Ryzen 7 7445HS (6 núcleos, 12 hilos lógicos), 15,2 GB de RAM y Windows 11 Home.

## Referencias citadas

Estilo autor–año; formatos tal como en `reports/borrador_jose_antecedentes.md` (hay que unificarlos al ensamblar el informe).

- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *JMLR, 13*, 281–305. https://jmlr.org/papers/v13/bergstra12a.html
- Gad, A. F. (2024). PyGAD: An intuitive genetic algorithm Python library. *Multimedia Tools and Applications, 83*(20), 58029–58042. https://doi.org/10.1007/s11042-023-17167-y
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
