# Aporte de Luis al informe técnico (actualizado, 23/oct/2026)

Este texto cubre **Datos y preparación**, **Resultados** y **Discusión**. Las
cifras que siguen proceden exclusivamente del entrenamiento y de validación
cruzada por grupos. La tabla consolidada contiene 16 filas. El equipo eligió
la MLP optimizada con algoritmo genético y semilla 42 como modelo final según
la regla acordada. Dayron hará la evaluación única en la prueba reservada el
27/oct; sus resultados aún no figuran en este borrador.

## Datos y preparación

Se utilizó *Condition Monitoring of Hydraulic Systems*, publicado por Helwig,
Pignanelli y Schütze en el [repositorio UCI](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)
(DOI: [10.24432/C5CW21](https://doi.org/10.24432/C5CW21); licencia CC BY 4.0;
consultado por el equipo el 20 de octubre de 2026). El banco hidráulico registra
2.205 ciclos de carga de 60 segundos con 17 sensores de presión, caudal,
temperatura, potencia, vibración y eficiencia. La variable objetivo es la presión
del acumulador: 90 bar (cercana a falla total), 100 bar (muy reducida), 115 bar
(algo reducida) o 130 bar (óptima). Esas categorías se codificaron como 0, 1, 2
y 3, respectivamente.

Cada ciclo se representó mediante cuatro estadísticas de cada sensor: media,
desviación estándar poblacional (`ddof=0`), mínimo y máximo. Se obtuvieron así
68 variables numéricas por ciclo. La extracción se hace fila a fila, sin
consultar otras observaciones ni sus etiquetas. Se excluyeron de las entradas
las etiquetas de los otros componentes, `stable`, `cycle_id` y `group_id`:
ninguna de ellas es una medición de sensor que deba usar el clasificador.

La partición común, almacenada en `split.csv`, separa **1.827 ciclos para
entrenamiento** y **378 para prueba**. Los ciclos con la misma combinación de
estados de los cuatro componentes reciben el mismo `group_id` y permanecen
juntos en una partición. Esto evita que configuraciones casi idénticas aparezcan
a ambos lados. Dentro del entrenamiento se usó `StratifiedGroupKFold` con cinco
pliegues y semilla 42. El conjunto de prueba no se consultó durante la
preparación, la exploración ni la búsqueda de hiperparámetros.

Los archivos originales contienen una serie de mediciones **dentro** de cada
ciclo de 60 segundos. Para esta tarea, cada ciclo se resume en una fila de 68
estadísticos; el clasificador no recibe el orden de los ciclos ni variables
rezagadas y no intenta pronosticar el siguiente instante. El PDF de la
asignatura pide separación temporal cuando se trabaja con series temporales.
El equipo acordó mantener `split.csv` por grupos porque aquí se evalúa la
clasificación de ciclos bajo **combinaciones de estados de componentes no
vistas**. La misma combinación puede aparecer en bloques temporales distintos;
separar por orden de los ciclos o por bloque permitiría que apareciera en
entrenamiento y prueba. Este diseño comprueba generalización entre
configuraciones, pero no mide explícitamente el desempeño en ciclos futuros de
la misma instalación.

En entrenamiento se encontraron 686 ciclos de 90 bar, 317 de 100 bar, 317 de
115 bar y 507 de 130 bar. No hubo valores nulos, no finitos ni filas de sensores
idénticas. Cuatro columnas (`PS2_min`, `PS3_min`, `FS1_min` y `SE_min`) fueron
constantes. No se justificó eliminar ciclos. Se detectaron 337 pares de
variables con correlación absoluta superior a 0,95. Por ello, cada modelo usa
un `Pipeline` con `StandardScaler`, eliminación de columnas constantes y un
filtro de redundancia por correlación ajustado solo con el entrenamiento de cada
pliegue. El umbral inicial `|r| > 0,995` se eligió tras comparar umbrales solo
en entrenamiento: `0,95` eliminaba demasiadas variables y reducía el F1 de la
MLP provisional. En el entrenamiento completo, el umbral de 0,995 conserva 41
de las 68 variables; la selección se vuelve a ajustar en cada pliegue.

Los 111 grupos presentes en entrenamiento tienen tamaños muy desiguales. Los
tres mayores contienen 230, 221 y 221 ciclos: **672 en conjunto, el 36,8 %**
del entrenamiento. Cada uno ocupa más de la mitad de uno de los tres primeros
pliegues de validación. Por ello se presentan resultados por pliegue además de
la media y dispersión, y se evita interpretar una única cifra como desempeño
uniforme en todas las configuraciones. El análisis de distribuciones señaló a
`PS5_max`, `PS3_mean` y `PS1_mean` como variables de presión relativamente poco
redundantes que pueden ayudar a explicar tendencias; sus distribuciones se
solapan entre clases, de modo que no constituyen reglas diagnósticas por sí
solas.

## Resultados de validación cruzada

Se compararon dos líneas base, KNN, SVM, Random Forest y MLP con el mismo
`split.csv`, cinco pliegues por grupos y el `Pipeline` compartido. KNN exploró
36 combinaciones, SVM RBF 30 y Random Forest 27. Para la MLP se compararon la
configuración de referencia, `RandomizedSearchCV` y un algoritmo genético (GA)
con **100 evaluaciones de CV por método y corrida**, repetidos con las semillas
42, 43 y 44. En todos los casos la métrica principal fue F1 macro. KNN eligió
11 vecinos, distancia Manhattan y ponderación por distancia; SVM eligió
`C = 45,71` y `gamma = 0,02481`; RF ajustado eligió 300 árboles, profundidad
sin límite y dos muestras como mínimo por hoja. La MLP de referencia utiliza
sus parámetros por defecto salvo `max_iter = 4000` y la semilla.

| Modelo | F1 macro CV (media ± DE) | Exactitud CV (media ± DE) | ROC-AUC OvR macro CV (media ± DE) |
|---|---:|---:|---:|
| Dummy (clase más frecuente) | 0,129 ± 0,039 | 0,363 ± 0,146 | 0,500 ± 0,000 |
| Dummy (estratificado) | 0,250 ± 0,028 | 0,278 ± 0,026 | 0,510 ± 0,017 |
| KNN ajustado | 0,528 ± 0,107 | 0,516 ± 0,149 | 0,792 ± 0,085 |
| SVM RBF ajustada | 0,767 ± 0,087 | 0,766 ± 0,092 | 0,938 ± 0,046 |
| Random Forest por defecto | 0,705 ± 0,132 | 0,694 ± 0,170 | 0,874 ± 0,086 |
| Random Forest ajustado | 0,715 ± 0,140 | 0,704 ± 0,177 | 0,870 ± 0,094 |
| MLP de referencia (semilla 42) | 0,725 ± 0,089 | 0,726 ± 0,128 | 0,916 ± 0,045 |
| MLP + búsqueda aleatoria (semilla 42) | 0,808 ± 0,077 | 0,811 ± 0,094 | 0,955 ± 0,029 |
| **MLP + GA (semilla 42)** | **0,812 ± 0,054** | **0,821 ± 0,062** | **0,955 ± 0,025** |

La tabla íntegra de **16 filas** está en `reports/resultados.csv`, formada a
partir de los archivos individuales de cada integrante. La MLP por defecto de
Dayron y la «MLP defecto (semilla 42)» de José son **la misma configuración y
los mismos cinco resultados**, por lo que aparecen una sola vez en la tabla
resumida y no se cuentan como dos modelos ni como corridas independientes.
Las otras seis filas de MLP corresponden a las semillas 43 y 44 de los tres
métodos. Esta tabla presenta CV; todavía no contiene resultados de prueba.

| Método MLP | F1 macro medio de CV, promedio de tres corridas ± DE entre corridas |
|---|---:|
| Referencia | 0,7254 ± 0,0049 |
| Búsqueda aleatoria | 0,8061 ± 0,0021 |
| Algoritmo genético | 0,8093 ± 0,0047 |

Los tiempos de `reports/resultados.csv` documentan el ajuste por pliegue de
la configuración seleccionada y la inferencia de `predict_proba` por lote de
validación; **no** incluyen el coste completo de las búsquedas. Los
experimentos de los integrantes cambiaron el paralelismo, por lo que esos
tiempos no son directamente comparables entre métodos y no sustentan una
conclusión de mayor velocidad. Las métricas por pliegue, clase y matriz de
confusión constan en los archivos `reports/pliegues_*.csv`,
`reports/metricas_clase_*.csv` y `reports/matrices_confusion_*.csv` y en las
figuras de `reports/figuras/`.

Las predicciones fuera de pliegue muestran un error importante para la
seguridad: de los 686 ciclos reales de 90 bar, KNN etiquetó 180 correctamente,
SVM 472 y RF ajustado 358. La **MLP+GA de semilla 42 identificó 551** (recall
0,803; F1 de esa clase 0,871). Esta última confundió solo **2 de 686 ciclos de
90 bar con 130 bar**; SVM confundió 69 y RF ajustado 213. Los otros 133 ciclos
de 90 bar errados por la MLP+GA se asignaron a 100 bar (60) o 115 bar (73).
Son predicciones de CV sobre entrenamiento, no la evaluación reservada.

## Selección, confianza y explicación

El equipo eligió **MLP+GA (semilla 42)** por el F1 macro medio de CV más alto
de la tabla (0,8123 ± 0,0543). Usa una capa de 64 neuronas, activación `tanh`,
`alpha = 0,003513`, tasa de aprendizaje inicial `0,002670` y
`max_iter = 4000`. La diferencia con otras configuraciones optimizadas es
pequeña; la elección sigue la regla del contrato y no demuestra por sí sola
que el GA sea superior como método de búsqueda. Dayron reentrenará ese
pipeline con todo el entrenamiento y hará la única evaluación en prueba el
27/oct. Esta evaluación aún no se ha efectuado.

José calculó el umbral de baja confianza con probabilidades **fuera de
pliegue**, siempre sobre entrenamiento. El umbral **0,87** es el menor de su
rejilla que logra al menos 90 % de exactitud entre predicciones aceptadas:
acepta **1.206 de 1.827 ciclos (66,0 %)** con exactitud **90,1 %**, y marca
621 (34,0 %) para revisión. Sin umbral la exactitud de estas predicciones es
82,9 %. La estimación del 90,1 % es frágil: el intervalo de remuestreo por
grupos va de 84,5 a 94,6 % y la exactitud de los aceptados varía de 78,9 a
97,9 % entre pliegues. El umbral se seleccionó y midió con los mismos datos
de entrenamiento, de modo que puede ser optimista. Las probabilidades de la
MLP no están calibradas: 0,87 es un corte operativo, no una garantía de acierto.

La explicación difusa de José usa `PS5_max`, `PS3_mean` y `PS1_mean`, elegidas
tras el EDA. Los percentiles 25, 50 y 75 del entrenamiento definen grados de
presión baja, media y alta. Si las tres variables están bajas, la regla apunta
a 130 bar; si alguna está alta, apunta a **menos de 130 bar**. Valores
intermedios no dan una pista clara. Las variables no son redundantes entre sí
(|r| máximo 0,76), pero las clases 90, 100 y 115 bar se solapan. Por ello la
regla no distingue esos tres niveles ni explica cómo decide la MLP, que usa
68 variables. Si la pista y la predicción discrepan, la interfaz debe invitar
a revisar el ciclo sin cambiar la clase predicha. Tanto los percentiles como
la comprobación de las reglas provienen de entrenamiento, sin usar prueba.

## Discusión

La SVM obtuvo mejor F1 macro CV que KNN, RF y la MLP de referencia, pero las
MLP optimizadas la superaron. La mejora del RF ajustado frente a su referencia
fue 0,0102 en F1 macro medio, pequeña respecto a las desviaciones entre
pliegues (0,132 y 0,140). El modelo elegido, MLP+GA de semilla 42, supera el
criterio (a): su F1 macro CV de 0,8123 está por encima de la mejor línea base,
Dummy estratificado, 0,2498 ± 0,0280, incluso al exigir más de dos
desviaciones de distancia (umbral aproximado 0,306). El criterio (b), que
contrasta CV con prueba, **sigue pendiente**. Las búsquedas de hiperparámetros
y la descripción posterior usan los mismos pliegues; estos F1 de CV pueden
estar favorecidos por la selección.

En el criterio (c), el GA promedió 0,8093 en las tres corridas frente a 0,8061
de la búsqueda aleatoria: ventaja de **0,0032**, menor que la variación entre
pliegues de CV (del orden de 0,06 a 0,08). Con el presupuesto común de 100
evaluaciones por corrida **no se observó una mejora clara** del GA. Tampoco
se puede afirmar equivalencia estadística con solo tres corridas por método.

El error más delicado es presentar como menos grave un acumulador de 90 bar.
La MLP+GA elegida redujo la confusión de 90 con 130 bar a dos casos en las
predicciones fuera de pliegue, pero no eliminó todos los errores: 135 de los
686 ciclos de 90 bar quedaron en otra clase. Un técnico podría retrasar una
inspección necesaria si interpreta la predicción como diagnóstico definitivo.
El prototipo debe mostrar probabilidades, marca de baja confianza y un aviso
de apoyo a la decisión. Aún no se ha medido si el umbral 0,87 marca justamente
los dos errores extremos; una probabilidad alta tampoco garantiza corrección.
En el otro sentido, una falsa alarma puede causar una revisión innecesaria.

Las cifras publicadas sobre este conjunto de datos no son comparables de
manera directa: varios trabajos estudian componentes distintos, resumen las
señales de otra forma o emplean repartos aleatorios o insuficientemente
descritos (véase el borrador de antecedentes de José). Nuestro reparto por
combinaciones completas exige generalizar a configuraciones no vistas y puede
dar resultados más bajos que una partición aleatoria por ciclos cercanos.

Las limitaciones principales son el origen en un único banco de laboratorio,
la coexistencia de fallas de otros componentes, la concentración de ciclos en
tres grupos grandes y el solapamiento de mediciones entre niveles de presión.
Las variables virtuales de eficiencia incluidas entre los sensores requieren
verificar su disponibilidad si se quisiera usar el prototipo fuera del banco.
El rendimiento en otra instalación no se conoce. Una validación cronológica
para predecir ciclos futuros respondería a otra pregunta y no se realizó.

### Pendiente para integrar la versión final

1. Incorporar la evaluación **única** en prueba que realizará Dayron el
   27/oct, por separado de CV; analizar el criterio (b), las métricas por clase
   y la confusión de 90 con 130 bar. No volver a escoger hiperparámetros con
   esos resultados.
2. Verificar las capturas del prototipo con el modelo final, la marca de baja
   confianza y la explicación difusa, incluidos casos de entrada inválida.
3. Integrar este aporte con el marco teórico de José y las conclusiones y
   referencias finales del equipo; preparar la presentación de máximo diez
   diapositivas con las cifras de prueba cuando estén disponibles.

### Referencia del conjunto de datos

Helwig, N., Pignanelli, E., & Schütze, A. (2015). *Condition monitoring of
hydraulic systems* [Conjunto de datos]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CW21

### Registro para el anexo de uso de IA

El 22/oct/2026 Luis utilizó Codex como apoyo para implementar y revisar el
`Pipeline`, preparar y ejecutar los notebooks de EDA y KNN/SVM, verificar sus
tablas y gráficas y redactar la primera versión. El 23/oct/2026 lo usó para
contrastar los CSV y notebooks ya publicados por José y Dayron e integrar sus
resultados de CV, el análisis de confianza y la explicación difusa en el
borrador. Los números proceden de los experimentos del equipo, no de la IA.
Luis es responsable de revisar las interpretaciones y el texto final.
