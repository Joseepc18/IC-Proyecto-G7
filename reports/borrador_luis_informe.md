# Aporte de Luis al informe técnico (borrador, 22/oct/2026)

Este texto cubre **Datos y preparación**, **Resultados** y **Discusión**. Se basa
exclusivamente en entrenamiento y validación cruzada por grupos. La tabla todavía
debe incorporar Random Forest, MLP y la optimización genética; la prueba reservada
se evaluará una sola vez después de seleccionar el modelo final con el equipo.

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

## Resultados de modelos clásicos (parciales)

Se compararon dos clasificadores de referencia y se ajustaron KNN y SVM con
F1 macro como objetivo de búsqueda. Para KNN se evaluaron 36 combinaciones de
número de vecinos, ponderación y distancia. Para SVM con núcleo RBF se
muestrearon 30 combinaciones de `C` y `gamma` en escala logarítmica. Todas las
búsquedas usaron los cinco pliegues por grupos y el mismo preprocesamiento.
Los parámetros elegidos fueron **KNN: 11 vecinos, distancia Manhattan y
ponderación por distancia**; **SVM: `C = 45,71` y `gamma = 0,02481`**.

| Modelo | F1 macro CV (media ± DE) | Exactitud CV (media ± DE) | ROC-AUC macro CV (media ± DE) |
|---|---:|---:|---:|
| Dummy (clase más frecuente) | 0,129 ± 0,039 | 0,363 ± 0,146 | 0,500 ± 0,000 |
| Dummy (estratificado) | 0,250 ± 0,028 | 0,278 ± 0,026 | 0,510 ± 0,017 |
| KNN ajustado | 0,528 ± 0,107 | 0,516 ± 0,149 | 0,792 ± 0,085 |
| SVM RBF ajustada | 0,767 ± 0,087 | 0,766 ± 0,092 | 0,938 ± 0,046 |

La tabla completa, con hiperparámetros y tiempos de entrenamiento e inferencia,
se genera en `reports/resultados.csv`; contiene por ahora solo los modelos ya
evaluados. `t_inferencia` mide el tiempo de `predict_proba` por lote de
validación, en segundos, sin incluir el cálculo de las métricas. En esta
ejecución, el tiempo medio de entrenamiento por pliegue fue cercano a 0,01 s
para KNN y 0,19 s para SVM; el de inferencia fue cercano a 0,01 s para ambos.
Los valores dependen del equipo y de su carga de trabajo.

La matriz de confusión con predicciones fuera de pliegue muestra que KNN
identifica solo 180 de los 686 ciclos de 90 bar (recall 0,262), mientras que la
SVM identifica 472 (recall 0,688). Para la SVM, los recall de 100, 115 y 130
bar son 0,861, 0,754 y 0,840. La precisión, el recall y el F1 de cada clase
se encuentran en `reports/metricas_clase_luis.csv` y las matrices en
`reports/figuras/matrices_confusion_luis.png`. Los cinco resultados por pliegue
se conservan en `reports/pliegues_luis.csv`.

## Discusión (parcial)

Entre los dos modelos clásicos ajustados, la SVM presenta mayor F1 macro medio
y mayor ROC-AUC que KNN. Esto **no** decide todavía el modelo final: faltan los
resultados de Random Forest, MLP y la optimización genética, que deben
compararse con el mismo criterio y presupuesto acordados por el equipo. Además,
los hiperparámetros se eligieron y se describieron con los mismos pliegues de
entrenamiento; las cifras de CV pueden ser optimistas por selección de modelo.
La comprobación independiente será la evaluación única sobre la prueba
reservada, una vez tomada la decisión conjunta.

Para el criterio de éxito (c), las tres corridas del GA se compararán con las
tres de búsqueda aleatoria bajo el mismo presupuesto de evaluaciones y con la
dispersión de la validación cruzada. Solo se afirmará una mejora si la
diferencia supera el umbral fijado por el contrato. En caso contrario se
informará que **no se observó una mejora clara**, sin afirmar equivalencia
estadística. Aún no se dispone de esos resultados.

El error más delicado es presentar como menos grave un acumulador de 90 bar.
En las predicciones fuera de pliegue, la SVM asignó **69 de esos 686 ciclos a
130 bar** (10,1 %), la clase óptima. Un técnico podría retrasar una inspección
necesaria si interpretara esa salida como diagnóstico definitivo. El prototipo
debe mostrar probabilidades, una señal de baja confianza y una advertencia de
que se trata de apoyo a la decisión. También debe comprobarse específicamente
si la marca de confianza detecta este tipo de error; una probabilidad alta no
garantiza que la predicción sea correcta ni calibrada. En el otro sentido, una
falsa alarma puede causar una revisión innecesaria y tiempo perdido.

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

1. Incorporar filas de Random Forest, MLP y GA de Dayron y José a la tabla
   única, con métricas por clase, matrices, cinco pliegues y tiempos
   comparables.
2. Seleccionar entre los tres integrantes **un solo modelo** por F1 macro,
   estabilidad, complejidad, inferencia y errores relevantes de 90 bar.
3. Tras la reunión acordada, añadir la evaluación **única** en prueba y
   discutir cualquier caída respecto a CV.
4. Integrar la explicación difusa o análisis de confianza de José, las
   implicaciones éticas y la conclusión conjunta.

### Referencia del conjunto de datos

Helwig, N., Pignanelli, E., & Schütze, A. (2015). *Condition monitoring of
hydraulic systems* [Conjunto de datos]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CW21

### Registro para el anexo de uso de IA

El 22/oct/2026 Luis utilizó Codex como apoyo para implementar y revisar el
`Pipeline`, preparar y ejecutar los notebooks de EDA y KNN/SVM, verificar las
tablas y gráficas y redactar este borrador. Los resultados numéricos proceden
de ejecuciones reproducibles sobre los datos de entrenamiento, no de la IA.
Luis es el autor y responsable de revisar e interpretar estos aportes.
