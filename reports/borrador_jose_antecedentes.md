# Aporte de José al informe técnico (versión pulida, 23/oct/2026)

Este texto cubre **Antecedentes**, **Marco teórico** y una **propuesta de problema, preguntas, objetivos e hipótesis**. Incluye la tabla de fuentes con enfoque, ventajas, limitaciones y métricas, y una lista única de referencias en APA 7. Los DOI se comprobaron con Crossref el 21, 22 y 23/oct; los de arXiv y del conjunto de datos de la UCI no están en Crossref y abren en doi.org, y Bergstra y Bengio (2012) y Pedregosa et al. (2011) no tienen DOI, por lo que se enlaza su página en JMLR. Lo que se dice de cada fuente procede de su resumen o de su texto en línea; la sección final indica qué no está verificado. No usa el conjunto de prueba. Los resultados del equipo no se discuten aquí, sino en Resultados y Discusión; las cifras del proyecto proceden de `reports/resultados.csv` y del borrador de Metodología.

## Antecedentes

Estilo de cita: autor–año (APA 7). Las referencias completas están en la lista única de «Referencias», al final del Marco teórico. Las tres obras de Helwig, Pignanelli y Schütze de 2015 se distinguen con letras, en el orden alfabético de sus títulos: 2015a (artículo de la I2MTC), 2015b (conjunto de datos) y 2015c (artículo de SENSOR).

El banco de pruebas hidráulico que usa este trabajo fue desarrollado en ZeMA (Saarbrücken) y publicado por Helwig, Pignanelli y Schütze (2015b) como conjunto de datos abierto en el repositorio de la UCI. Repite ciclos de carga constantes de 60 s y registra 17 sensores mientras varía a propósito el estado de cuatro componentes (enfriador, válvula, bomba y acumulador), con 2.205 ciclos en total. Sus autores lo emplearon para monitorear la condición del sistema con estadística multivariada (Helwig et al., 2015a) y para detectar y compensar fallas de los propios sensores, con extracción, selección y clasificación automáticas de variables mediante análisis discriminante lineal (Helwig et al., 2015c).

Desde entonces el conjunto se ha usado para comparar enfoques muy distintos. Alenany et al. (2021) reducen el número de sensores y las variables con Random Forest, y reportan un entrenamiento entre 25,7 % y 36,4 % más rápido y un modelo entre 70,3 % y 85,5 % más pequeño, manteniendo la precisión. Khan et al. (2023) convierten las series en espectrogramas y las clasifican con una red residual (ResNet-18), pero solo para el estado del enfriador y con 440 espectrogramas; reportan una exactitud de prueba del 96 %. Yildirim et al. (2025) proyectan los datos crudos con análisis de componentes principales funcionales multivariado y clasifican con Random Forest los cuatro componentes: la exactitud global es del 96,2 %, y el acumulador es el componente más difícil, con 89,6 %, frente a 95,2 % de la válvula y 100 % del enfriador y la bomba. Goodarzi et al. (2023), en un preprint, comparan una línea base clásica con redes convolucionales para el estado del acumulador en cuatro clases, el mismo objetivo de este trabajo; la línea base alcanza un error cercano al 1 %, y una red convolucional con los 17 sensores llega al 20,5 %.

Estos trabajos comparten limitaciones relevantes para nosotros. Primero, reportan sobre todo exactitud o tasa de error (Khan et al. añaden precisión, recall y F-score), y en lo que se pudo leer no se menciona un criterio de éxito fijado de antemano frente a una línea base. Segundo, el reparto entre entrenamiento y prueba es aleatorio en el trabajo que lo describe con detalle (Goodarzi et al., 2023: 70/10/20) y no se precisa en los demás (Khan et al., 2023; Yildirim et al., 2025, que solo indican 70/30). En este banco los ciclos vienen en bloques consecutivos con la misma configuración de componentes (194 bloques, 144 combinaciones distintas; mediana de 10 ciclos por combinación), por lo que un reparto aleatorio puede colocar ciclos casi idénticos a ambos lados y sobreestimar el desempeño. Por eso las cifras de la literatura no son comparables directamente con las de este trabajo, que parte por combinaciones completas de componentes y usa el F1 macro como métrica principal.

En cuanto a la optimización de hiperparámetros, Bergstra y Bengio (2012) mostraron que muestrear al azar el mismo espacio de búsqueda puede igualar o superar a la búsqueda en rejilla con mucho menos cómputo, porque pocos hiperparámetros influyen de verdad en cada conjunto de datos, y propusieron la búsqueda aleatoria como línea de base para evaluar métodos adaptativos. Los algoritmos genéticos son una de esas alternativas: Chiroma et al. (2017) revisan su uso para optimizar redes neuronales, organizado por dominio de aplicación, diseño de la red y valores de los operadores genéticos. La evidencia sobre esa comparación es mixta. Valsecchi et al. (2021) compararon rejilla, búsqueda aleatoria, TPE y algoritmo genético para redes neuronales multitarea en tres conjuntos de química (QSAR), con 100 arquitecturas por corrida y 10 corridas independientes: el algoritmo genético y TPE superaron a la búsqueda aleatoria de forma significativa en validación cruzada, y en dos de los tres conjuntos en la prueba externa. En cambio, el argumento de Bergstra y Bengio (2012) sugiere que la búsqueda aleatoria es difícil de superar cuando pocos hiperparámetros importan. El-Hassani et al. (2024) aplican un algoritmo genético a los hiperparámetros de una MLP en conjuntos de la UCI, aunque en lo leído no lo comparan con la búsqueda aleatoria. En la búsqueda realizada no encontramos una comparación de este tipo con una MLP y datos de sensores; este trabajo la evalúa con el mismo presupuesto de 100 evaluaciones y tres repeticiones por método, sin presuponer que el algoritmo genético gane.

Por último, el razonamiento aproximado aporta una vía de explicación. Filo (2023) revisa la lógica difusa en sistemas hidráulicos y neumáticos y encuentra el control y el diagnóstico de fallas entre sus áreas principales de aplicación, con atención a las funciones de pertenencia, los operadores y los métodos de inferencia. Cummins et al. (2024), en una revisión de la explicabilidad en mantenimiento predictivo que sigue las guías PRISMA, subrayan que los operadores necesitan confiar en estos sistemas cuando las decisiones son de alto riesgo. No encontramos trabajos que apliquen lógica difusa a este conjunto de datos; este trabajo la usa de forma acotada para describir tendencias en unas pocas variables de presión junto a la predicción del modelo seleccionado y una marca de baja confianza basada en probabilidades.

**Resumen de la brecha.** En los trabajos revisados (una búsqueda no sistemática, en parte basada en resúmenes), la literatura sobre este banco reporta altas exactitudes con repartos aleatorios o no especificados y no fija de antemano un criterio de éxito; no encontramos una comparación de algoritmos genéticos con búsqueda aleatoria bajo un presupuesto igual (lo más cercano es Valsecchi et al., en otro dominio), ni una explicación o medida de confianza dirigida al técnico. Este trabajo aborda esas tres brechas para el estado del acumulador, sin afirmar que no existan en la literatura no revisada.

## Fuentes consultadas

### Dataset y mantenimiento predictivo hidráulico

| # | Fuente (ver Referencias) | Enfoque | Ventajas | Limitaciones | Métricas |
|---|---|---|---|---|---|
| 1 | Helwig et al. (2015a) | Artículo original del banco de pruebas; estadística multivariada, ciclo fijo de 60 s | Fuente primaria del dataset | Resumen cerrado en IEEE: no se verificó nada más | No verificado |
| 2 | Helwig et al. (2015c) | Extracción, selección y clasificación automáticas (LDA); detección y compensación de fallas de sensores | Tolera la falla de hasta 5 sensores | El resumen no da cifras | No reportadas en el resumen |
| 3 | Alenany et al. (2021) | Random Forest con 6 variables extraídas y sensores reducidos | Entrenamiento 25,7–36,4 % más rápido; modelo 70,3–85,5 % más pequeño | Cita el >99 % de la literatura sin medirlo; el resumen no da tipo de validación | Tiempo y tamaño en disco |
| 4 | Khan et al. (2023) | Espectrogramas (FFT) + ResNet-18, solo `cooler` | Aprendizaje profundo sobre el mismo dataset | 3 clases, 440 espectrogramas; reparto no especificado (resumen: "30 % probado y validado"); sin CV; cifras inconsistentes entre secciones (verificado en el texto completo) | Exactitud 96 %, precisión 96 %, recall 95 % (prueba) |
| 5 | Yildirim et al. (2025) | MFPCA + Random Forest sobre datos crudos, 4 componentes | Reciente (15/ene/2025, según la página de la revista); resultados por componente | Reparto 70/30 sin decir si es por grupos; sin F1 | Exactitud: global 96,2 %; `accumulator` 89,6 %; valve 95,2 %; cooler y pump 100 % |
| 6 | Goodarzi et al. (2023), preprint | CNN (fusión temprana y tardía) vs línea base clásica; objetivo `accumulator`, 4 clases | Mismo objetivo que el nuestro | Preprint; reparto aleatorio 70/10/20 (verificado en el texto del PDF); búsqueda de hiperparámetros limitada a 50 iteraciones | Error de prueba: línea base ~1 %; CNN con 17 sensores 20,5 % |

### Optimización, lógica difusa y explicabilidad

| # | Fuente (ver Referencias) | Enfoque | Ventajas | Limitaciones | Métricas |
|---|---|---|---|---|---|
| 7 | Chiroma et al. (2017) | Revisión de redes neuronales optimizadas con GA, por dominio, tipo de red y operadores | Justifica operadores y configuración del GA | De 2017; solo se leyó el resumen | No aplica |
| 8 | Bergstra y Bengio (2012) | Búsqueda aleatoria vs rejilla | Justifica `RandomizedSearchCV` como comparación; propone la búsqueda aleatoria como línea de base para métodos adaptativos | De 2012; sin DOI; compara con rejilla y búsqueda manual, no incluye GA (verificado en el resumen) | Error de prueba vs número de pruebas |
| 9 | Filo (2023) | Revisión de lógica difusa en control y diagnóstico: funciones de pertenencia, operadores, inferencia, desdifusificación | Reciente y específica de hidráulica | El resumen dice que la mayor parte de la investigación busca ahorro de energía y mejor desempeño operativo; el diagnóstico es solo una de las áreas (lectura del resumen) | No aplica |
| 10 | Cummins et al. (2024) | Revisión PRISMA de IA explicable en mantenimiento predictivo, incluye sistemas difusos | Reciente; respalda explicación y confianza | General, no hidráulica | No aplica |
| 11 | Valsecchi et al. (2021) | Compara búsqueda en rejilla, búsqueda aleatoria, TPE y algoritmo genético para redes neuronales multitarea en 3 conjuntos QSAR | Es el estudio más cercano al nuestro: GA y búsqueda aleatoria con 100 arquitecturas por corrida y 10 corridas independientes, y prueba t al 95 % | Otro dominio (química, QSAR) y red multitarea, no una MLP con datos de sensores; el GA se resume con su población final y la búsqueda aleatoria con sus 10 mejores de 100, lo que hace la comparación algo asimétrica | "Non-error rate" en CV de 3 pliegues y en prueba externa; tiempo de cómputo. GA y TPE mejores que la búsqueda aleatoria en los 3 conjuntos en CV (significativo) y en 2 de 3 en la prueba externa (verificado en el texto completo) |
| 12 | El-Hassani et al. (2024) | GA con codificación real para optimizar arquitectura e hiperparámetros de una MLP (tasa de aprendizaje, activación, regularización), 4 conjuntos de la UCI | Reciente; GA aplicado justo a hiperparámetros de MLP | Solo se leyó el resumen: no se vio comparación con búsqueda aleatoria | Reporta mejora de exactitud y menor complejidad (sin cifras en el resumen) |

Conjunto de datos: Helwig et al. (2015b), disponible en la UCI con licencia CC BY 4.0 (referencia completa en «Referencias»).

### Hallazgos para el informe
- Goodarzi et al. (fuente 6) usan una partición aleatoria 70/10/20; Yildirim et al. (fuente 5) informan una partición 70/30 sin especificar el procedimiento (aleatorio, estratificado o por grupos). Sus exactitudes no son comparables con nuestro F1 macro con partición por grupos.
- `accumulator` es el objetivo más difícil en la fuente 5 (89,6 % frente a 95–100 % del resto).
- GA frente a búsqueda aleatoria: la evidencia previa es mixta. Valsecchi et al. (2021, fila 11) encuentran al GA mejor con presupuesto igual de 100 evaluaciones, pero en otro dominio (QSAR, redes multitarea); Bergstra y Bengio (2012) indican que la búsqueda aleatoria es una línea de base fuerte cuando pocos hiperparámetros importan. No apareció un estudio con una MLP y datos de sensores. La búsqueda fue de varias consultas en buscadores y Crossref, no una revisión sistemática.

## Marco teórico

### Red neuronal multicapa (MLP)
Una red neuronal multicapa (MLP) es un modelo supervisado formado por una capa de entrada, una o más capas ocultas y una capa de salida. Cada neurona calcula una combinación lineal de sus entradas y le aplica una función de activación no lineal, como ReLU o tangente hiperbólica. Sin esa no linealidad, apilar capas equivaldría a una sola transformación lineal. Cybenko (1989) demostró que una red con una sola capa oculta y una activación sigmoidal continua puede aproximar, con la precisión que se quiera, cualquier función continua de n variables definida sobre el hipercubo unitario (un dominio acotado). Ese resultado justifica su capacidad expresiva, pero no garantiza que el entrenamiento la encuentre.

En clasificación multiclase, la capa de salida usa softmax: devuelve una probabilidad por clase que suma 1. La red se entrena minimizando la entropía cruzada (log-loss). Los gradientes se calculan por retropropagación (Rumelhart et al., 1986) y los pesos se actualizan con un optimizador. En el proyecto se usa Adam (Kingma y Ba, 2014), el optimizador por defecto de `MLPClassifier` (Pedregosa et al., 2011).

**Relación con el problema**
- Entrada: 68 variables por ciclo, estandarizadas dentro del `Pipeline` (las MLP son sensibles a la escala). Salida: 4 probabilidades (90, 100, 115 y 130 bar).
- La MLP es el modelo seleccionado (MLP + GA; ver Metodología) y sus probabilidades softmax alimentan la marca de baja confianza del prototipo. Softmax no garantiza probabilidades calibradas: por eso el umbral se calibra con predicciones fuera de pliegue.
- Hiperparámetros que optimiza el GA: `hidden_layer_sizes` (capacidad), `alpha` (penalización L2), `learning_rate_init` (paso de Adam) y `activation` (relu o tanh).

### Algoritmo genético (GA)
Un algoritmo genético es una metaheurística de optimización inspirada en la evolución biológica; Katoch et al. (2021) revisan sus variantes, sus operadores genéticos y sus dominios de aplicación. Mantiene una población de soluciones candidatas (individuos), cada una codificada como un vector de genes. En cada generación evalúa la aptitud con una función objetivo y aplica tres operadores: **selección** (favorece a los individuos de mayor aptitud como padres), **cruce** (combina genes de dos padres) y **mutación** (altera al azar algunos genes para mantener diversidad).

Es una búsqueda estocástica global: no necesita gradientes y admite variables enteras, continuas y categóricas a la vez, por lo que se usa para optimizar redes neuronales (Chiroma et al., 2017). No garantiza el óptimo y su resultado varía entre corridas; de ahí las 3 repeticiones (semillas 42, 43 y 44).

**Relación con el problema**
- Individuo: `hidden_layer_sizes` (índice entre 5 opciones), `alpha` y `learning_rate_init` (continuos, escala logarítmica) y `activation`.
- Aptitud: F1 macro medio de la CV por grupos. Se fija el `random_state` de la MLP en cada evaluación; si no, la aptitud de un mismo individuo cambia entre llamadas y el GA persigue ruido.
- Implementación: PyGAD 3.9.2 (Gad, 2024). El tamaño de la población, el número de generaciones y los operadores concretos están en la Metodología.
- Comparación: `RandomizedSearchCV` es la referencia estándar (Bergstra y Bengio, 2012). El criterio (c) del contrato exige que el GA supere a esa búsqueda por más que la desviación entre pliegues de la CV; si no, se escribe "no se observó una mejora clara". La variación entre semillas (3 corridas por método) se informa por separado y no sustituye a ese criterio. La aplicación del criterio a los resultados está en la Metodología.

**Presupuesto de evaluaciones.** PyGAD no vuelve a evaluar a los padres que conserva, así que el número de evaluaciones no es `población × generaciones`. Una prueba con una aptitud de juguete (PyGAD 3.9.2) lo mostró:

| Configuración | Llamadas a la aptitud |
|---|---|
| población 10, 9 generaciones, `keep_elitism=1` (por defecto) | 91 |
| población 10, 9 generaciones, `keep_elitism=0`, `keep_parents=0` | 100 |
| población 20, 4 generaciones, `keep_elitism=1` | 96 |

Para que el GA y `RandomizedSearchCV` gasten lo mismo hay que contar las llamadas dentro de la función de aptitud y fijar el elitismo y los padres conservados. Quitar solo el elitismo no basta: con 2 padres conservados el conteo fue de 82 llamadas. La configuración usada y la comprobación de que cada corrida real gastó exactamente 100 evaluaciones están en la Metodología.

### Lógica difusa y manejo de incertidumbre
La lógica difusa generaliza la lógica clásica: en lugar de decir que un elemento pertenece o no a un conjunto, le asigna un **grado de pertenencia** entre 0 y 1 (Zadeh, 1965). Permite representar conceptos graduales como "presión baja" o "presión algo reducida", sin un corte rígido.

Los sistemas difusos de tipo Mamdani toman su nombre de un trabajo experimental en el que reglas heurísticas de un operador, expresadas en términos lingüísticos, se convirtieron en una estrategia de control automático para una máquina de vapor (Mamdani y Assilian, 1975). En general, estos sistemas convierten cada valor medido en grados de pertenencia a términos como «bajo», «medio» o «alto» y los combinan mediante reglas SI–ENTONCES, por ejemplo «SI la presión media es baja ENTONCES el nivel del acumulador es muy reducido». En esa formulación, la conjunción (Y) se interpreta como el mínimo de los grados de pertenencia, la disyunción (O) como el máximo y varias reglas se combinan también con el máximo (Mamdani y Assilian, 1975). Este trabajo usa el mismo criterio: mínimo cuando deben cumplirse las tres condiciones y máximo cuando basta con una, como se detalla en la Metodología. Su ventaja principal es que las reglas son legibles para una persona. Una revisión reciente de lógica difusa en sistemas hidráulicos (Filo, 2023) identifica el control y el diagnóstico de fallas entre sus áreas principales de aplicación.

**Relación con el problema**
- Dos tipos de incertidumbre: la probabilidad de softmax expresa la confianza del modelo en una clase; la lógica difusa expresa la vaguedad de la medida (108 bar no es nítidamente "100" ni "115"). Se complementan.
- Los niveles de presión (90, 100, 115 y 130 bar) son ordinales y se describen bien con términos lingüísticos.
- Papel en el prototipo: **descripción de tendencias**, no un quinto clasificador. Sobre unas pocas variables de presión se muestra qué reglas se activan y con qué grado, junto a la predicción del modelo seleccionado. Unas reglas independientes no explican por sí solas las decisiones del modelo: por eso se mide cuánto coinciden con él antes de presentarlas como explicación (resultados en la Metodología).
- Criterios de diseño: las funciones de pertenencia se definen solo con datos de entrenamiento, y la coincidencia de la explicación con la predicción del modelo seleccionado se mide con predicciones fuera de pliegue, nunca en prueba. Las variables, las funciones y las reglas concretas están en la Metodología.
- No se encontró una fuente que aplique lógica difusa a este conjunto de datos: el enlace con el problema es razonamiento propio.

### Referencias

Lista única del informe en APA 7, en orden alfabético (incluye las obras citadas en el borrador de Metodología). Donde no hay DOI se da la página del editor.

- Alenany, A., Helmi, A. M., & Nasef, B. M. (2021). Comprehensive analysis for sensor-based hydraulic system condition monitoring. *International Journal of Advanced Computer Science and Applications, 12*(6). https://doi.org/10.14569/IJACSA.2021.0120615
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research, 13*, 281–305. https://jmlr.org/papers/v13/bergstra12a.html
- Chiroma, H., Noor, A. S. M., Abdulkareem, S., Abubakar, A. I., Hermawan, A., Qin, H., Hamza, M. F., & Herawan, T. (2017). Neural networks optimization through genetic algorithm searches: A review. *Applied Mathematics & Information Sciences, 11*(6), 1543–1564. https://doi.org/10.18576/amis/110602
- Cummins, L., Sommers, A., Ramezani, S. B., Mittal, S., Jabour, J., Seale, M., & Rahimi, S. (2024). Explainable predictive maintenance: A survey of current methods, challenges and opportunities. *IEEE Access, 12*, 57574–57602. https://doi.org/10.1109/ACCESS.2024.3391130
- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals, and Systems, 2*(4), 303–314. https://doi.org/10.1007/BF02551274
- El-Hassani, F. Z., Amri, M., Joudar, N.-E., & Haddouch, K. (2024). A new optimization model for MLP hyperparameter tuning: Modeling and resolution by real-coded genetic algorithm. *Neural Processing Letters, 56*(2), Artículo 105. https://doi.org/10.1007/s11063-024-11578-0
- Filo, G. (2023). A review of fuzzy logic method development in hydraulic and pneumatic systems. *Energies, 16*(22), Artículo 7584. https://doi.org/10.3390/en16227584
- Gad, A. F. (2024). PyGAD: An intuitive genetic algorithm Python library. *Multimedia Tools and Applications, 83*(20), 58029–58042. https://doi.org/10.1007/s11042-023-17167-y
- Goodarzi, P., Robin, Y., Schütze, A., & Schneider, T. (2023). *Deep convolutional neural networks for cyclic sensor data* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2308.06987
- Helwig, N., Pignanelli, E., & Schütze, A. (2015a). Condition monitoring of a complex hydraulic system using multivariate statistics. En *2015 IEEE International Instrumentation and Measurement Technology Conference (I2MTC) Proceedings* (pp. 210–215). IEEE. https://doi.org/10.1109/I2MTC.2015.7151267
- Helwig, N., Pignanelli, E., & Schütze, A. (2015b). *Condition monitoring of hydraulic systems* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CW21
- Helwig, N., Pignanelli, E., & Schütze, A. (2015c). Detecting and compensating sensor faults in a hydraulic condition monitoring system. En *Proceedings SENSOR 2015* (pp. 641–646). AMA Service GmbH. https://doi.org/10.5162/sensor2015/D8.1
- Katoch, S., Chauhan, S. S., & Kumar, V. (2021). A review on genetic algorithm: Past, present, and future. *Multimedia Tools and Applications, 80*(5), 8091–8126. https://doi.org/10.1007/s11042-020-10139-6
- Khan, H. A., Bhatti, U., Kamal, K., Alkahtani, M., Abidi, M. H., & Mathavan, S. (2023). Fault classification for cooling system of hydraulic machinery using AI. *Sensors, 23*(16), Artículo 7152. https://doi.org/10.3390/s23167152
- Kingma, D. P., & Ba, J. (2014). *Adam: A method for stochastic optimization* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1412.6980
- Mamdani, E. H., & Assilian, S. (1975). An experiment in linguistic synthesis with a fuzzy logic controller. *International Journal of Man-Machine Studies, 7*(1), 1–13. https://doi.org/10.1016/S0020-7373(75)80002-2
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature, 323*(6088), 533–536. https://doi.org/10.1038/323533a0
- Valsecchi, C., Consonni, V., Todeschini, R., Orlandi, M. E., Gosetti, F., & Ballabio, D. (2021). Parsimonious optimization of multitask neural network hyperparameters. *Molecules, 26*(23), Artículo 7254. https://doi.org/10.3390/molecules26237254
- Yildirim, C., Franco-Pereira, A. M., & Lillo, R. E. (2025). Condition monitoring and multi-fault classification of hydraulic systems using multivariate functional data analysis. *Heliyon, 11*(1), Artículo e41251. https://doi.org/10.1016/j.heliyon.2024.e41251
- Zadeh, L. A. (1965). Fuzzy sets. *Information and Control, 8*(3), 338–353. https://doi.org/10.1016/S0019-9958(65)90241-X

## Propuesta de problema, preguntas, objetivos e hipótesis

**Formulación adoptada.** El problema, las preguntas P1 y P2, los objetivos y las hipótesis que siguen son la formulación del informe. Los criterios de éxito (a), (b) y (c) del contrato no cambian.

**Problema.** En una instalación hidráulica industrial, la pérdida de presión del acumulador degrada el funcionamiento y puede acabar en falla. En un banco de pruebas real la presión se fija a propósito en cuatro niveles (130, 115, 100 y 90 bar) mientras los demás componentes también varían. Se busca estimar ese nivel solo con las lecturas de los sensores de cada ciclo de 60 s. Hacerlo bien permitiría apoyar a los técnicos de mantenimiento a detectar la pérdida antes de la falla. La utilidad práctica no se prueba aquí: los datos vienen de un único banco de laboratorio.
- Cambios frente al texto del contrato (sección 1): "evita paradas y daños" pasa a "permitiría apoyar" (ningún dato del proyecto demuestra ese beneficio); "pierde presión de forma gradual" pasa a "niveles fijados a propósito", que es como se describe en el conjunto de datos.

**Objetivo general.** Comparar cuatro modelos de aprendizaje automático (KNN, SVM, Random Forest y MLP) para clasificar el nivel de presión del acumulador de un banco de pruebas hidráulico a partir de las lecturas de sus sensores, optimizar la MLP con un algoritmo genético y entregar un prototipo que devuelva la clase, sus probabilidades y una marca de baja confianza.

**Usuarios beneficiarios e impacto esperado.** El usuario previsto es el personal técnico y de mantenimiento de instalaciones hidráulicas; como usuarios secundarios están quienes diseñan o evalúan bancos de prueba y quienes revisan este trabajo. El impacto esperado es de apoyo, no de sustitución: ayudar a decidir qué acumuladores revisar primero cuando la presión parece reducida, y avisar cuando el modelo duda para que la decisión pase a una revisión humana. Este impacto no se demuestra aquí, porque el sistema no se probó fuera del banco de laboratorio.

**Implicaciones éticas.**
- **Datos.** El conjunto es público, procede de un banco de laboratorio y no contiene datos personales ni de individuos. Según la UCI se distribuye con licencia CC BY 4.0 y se cita a sus autores (Helwig et al., 2015b).
- **Errores y consecuencias.** El error más delicado es tomar un acumulador de 90 bar, el más degradado, por uno en mejor estado. En las predicciones fuera de pliegue, la MLP elegida confundió 2 de los 686 ciclos de 90 bar con 130 bar, pero otros 133 de esos ciclos los asignó a 100 o 115 bar (135 en total quedaron en otra clase). Una falsa alarma, en cambio, cuesta una revisión innecesaria. Ambos errores existen y no se eliminan.
- **Responsabilidad humana.** La predicción es una ayuda para decidir y no un diagnóstico. El umbral de baja confianza (0,87) es un punto de corte práctico: las probabilidades de la MLP no están calibradas y una probabilidad alta no garantiza que la predicción sea correcta.
- **Alcance y sesgo.** Los datos vienen de un único banco, con 144 combinaciones de estados de componentes. No se probó con otras instalaciones, sensores ni condiciones, así que no debe usarse para decisiones de seguridad sin una validación propia.
- **Transparencia.** Se fijan la semilla y las versiones, se separa la evaluación de prueba de la validación cruzada y se declara el uso de herramientas de IA generativa en el anexo.

**Preguntas** (la acordada, separada en dos porque tienen respuestas distintas)
- **P1.** ¿Con qué F1 macro pueden KNN, SVM, Random Forest y MLP identificar el nivel de presión a partir de un ciclo de 60 s, en combinaciones de estados de componentes no vistas en el entrenamiento?
- **P2.** Con el mismo presupuesto de 100 evaluaciones, ¿mejora un algoritmo genético a `RandomizedSearchCV` al optimizar la MLP?

**Objetivos específicos** (misma estructura, con lo medible explícito)
1. Extraer las 68 variables y partir por grupos de combinación, sin fuga de información.
2. Comparar los cuatro modelos con una línea base bajo el mismo `Pipeline` y la misma CV. Evaluar el GA frente a `RandomizedSearchCV` y a la MLP por defecto, con 3 corridas cada uno.
3. Construir un prototipo que devuelva la clase, las probabilidades, una marca de baja confianza y una explicación difusa sencilla de tendencias.

**Hipótesis** (falsables, alineadas con los criterios de éxito)
- **H1.** Los cuatro modelos superan a `Dummy(stratified)` por más de 2 desviaciones en F1 macro de CV.
- **H2 (exploratoria, post hoc).** Random Forest y MLP superan a KNN y SVM. Está motivada por mediciones sobre entrenamiento con modelos por defecto (≈ 0,72–0,75 frente a ≈ 0,49–0,53), así que no es una hipótesis fijada de antemano. Se mantiene tal como está y se evalúa con los resultados finales, sin reformularla para que encaje con lo observado: los resultados de la SVM ya ajustada (F1 macro 0,767 ± 0,087 en `reports/resultados.csv`) la contradicen para esa comparación, y así se informará. No se plantea que MLP supere a Random Forest: con las cifras medidas están empatados.
- **H3.** El GA supera a `RandomizedSearchCV` en F1 macro de CV por más que la desviación entre pliegues de la CV (criterio (c)); la variación entre semillas se informa por separado. La evidencia previa es mixta (Valsecchi et al., 2021; Bergstra y Bengio, 2012), así que el resultado está abierto y "no se observó una mejora clara" es una conclusión válida.
- **H4.** Las predicciones con probabilidad máxima sobre el umbral tienen mayor exactitud que las marcadas como baja confianza, medido con predicciones fuera de pliegue.

## Qué no está verificado
- Goodarzi (fila 6): reparto aleatorio 70/10/20, 50 iteraciones de búsqueda y errores de 1 %, 20,5 % y 1,7 % (PS1 sola) contrastados con el texto del PDF. Khan (fila 4) y Yildirim (fila 5): contrastados con el texto completo en XML de Europe PMC. Khan reporta en prueba exactitud 96 %, precisión 96 % (la conclusión dice 95 %), recall 95 % y F-score 95 %; los recuentos por clase de los 440 espectrogramas difieren entre secciones (146/146/148 frente a 146/148/146). Yildirim: "70 % entrenamiento y 30 % prueba", sin decir si es aleatorio, estratificado o por grupos, y sin validación cruzada para clasificar; usa los 2.205 ciclos, no solo los estables.
- Filas 1, 2, 3, 7, 8, 9 y 12: solo se leyó el resumen o la información del editor. Fila 10: no consta la lectura de su texto completo. Antes de citar una cifra de ellas, contrastarla con el PDF.
- Los textos completos de Cybenko (1989), Katoch et al. (2021) y Mamdani y Assilian (1975) se leyeron el 23/oct y respaldan lo que se afirma: Cybenko, que con una capa oculta y cualquier sigmoide continua las sumas finitas son densas en las funciones continuas del hipercubo unitario (Teorema 2); Katoch, que el algoritmo genético se inspira en la selección natural y que sus operadores son selección, cruce y mutación; Mamdani y Assilian, que la lógica difusa convierte reglas heurísticas de un operador en una estrategia de control automático (máquina de vapor) y que Y se interpreta como mínimo, O como máximo y las reglas se combinan con el máximo.
- Zadeh (1965), Rumelhart et al. (1986), Kingma y Ba (2014) y Pedregosa et al. (2011): se verificaron DOI o enlace y metadatos; lo que se dice de ellos es su aporte conocido y no se contrastó con su texto en esta revisión. Kingma y Ba figura con el año del preprint en arXiv (2014), no con el de la conferencia ICLR (2015).
- El conteo de 91, 82 y 96 llamadas de PyGAD se midió con una función de juguete, no con la MLP real. Que cada corrida real usó exactamente 100 evaluaciones consta en el notebook `03_metaheuristica` y en el borrador de Metodología.
- Que la hipótesis nula (GA ≈ búsqueda aleatoria) sea plausible es una inferencia propia: Bergstra y Bengio no prueban GA y Valsecchi et al. encuentran al GA mejor, en otro dominio.
- La licencia CC BY 4.0 del conjunto de datos viene del borrador anterior; la API de DataCite no la devolvió, así que no se reconfirmó el 23/oct.

## Registro para el anexo de uso de IA generativa
Para el contenido de este documento se usó un asistente de IA generativa (Claude Code, modelo Claude Sonnet 5.5) para buscar y verificar fuentes académicas, redactar borradores del marco teórico y de los antecedentes, proponer la formulación del problema y de las hipótesis, y escribir scripts de apoyo para el conteo de evaluaciones de PyGAD, usando solo datos de entrenamiento. El trabajo se hizo bajo la dirección de José, que indicó qué hacer en cada paso, abrió en el navegador varios DOI (MDPI, Springer, Heliyon) para confirmar autores y fechas, y aprobó los resultados. El resto del uso de IA en el proyecto lo declara cada integrante en su registro.
