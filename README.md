# Inteligencia Computacional · Grupo G7

## Tema

**Tema 7: industria.** Clasificación del nivel de presión de un acumulador
hidráulico a partir de las mediciones de 17 sensores.

## Problema del proyecto

La pérdida de presión del acumulador puede afectar el funcionamiento de un
sistema hidráulico. A partir de las lecturas de los sensores durante un ciclo
de 60 segundos, se busca estimar a qué categoría corresponde el acumulador:
**90, 100, 115 o 130 bar**, como apoyo para el personal de mantenimiento.

El análisis se limita al acumulador; no determina el estado de los demás
componentes ni el momento en que se detendrá la máquina.

**Bar** es una unidad de presión. En este proyecto, las categorías del acumulador
se interpretan así:

| Presión del acumulador | Estado que representa |
|---|---|
| **90 bar** | Cercano a falla total; es la categoría de mayor deterioro. |
| **100 bar** | Presión muy reducida. |
| **115 bar** | Presión algo reducida. |
| **130 bar** | Presión óptima; es la condición de referencia en buen estado. |

## Integrantes

- José Eduardo Ponce Carlos.
- Luis Alfredo Anchundia Cruz.
- Dayron Adrian Quiñónez Valencia.

## Estructura del proyecto

```text
app/
  app.py                       Flujo de la aplicación Streamlit
  assets/                      Estilos CSS y fondo SVG local
src/ic_g7/
  config.py                    Rutas, clases, semilla y columnas compartidas
  datos.py                     Carga de entrenamiento y prueba
  evaluacion.py                Validación cruzada y métricas
  pipeline.py                  Preparación común dentro de cada pliegue
  experimentos_dayron.py        Random Forest con búsqueda y MLP base
  modelo_final.py               Reentrenamiento y evaluación única reservada
  prototipo.py                 Validación del CSV y modelo Dummy
  interfaz/                    Componentes visuales y carga del tema
scripts/                       Descarga y preparación de los datos
data/
  raw/hydraulic/               Mediciones originales (no se suben a Git)
  processed/
    features.csv               Datos preparados (generado; no se sube a Git)
    split.csv                  Partición compartida de entrenamiento y prueba
notebooks/                     Espacio para análisis y experimentos
models/                        Destino del modelo final y las etiquetas
reports/                       Resultados y figuras
tests/                         Comprobaciones de datos, modelos y prototipo
requirements.txt               Dependencias del proyecto
pyproject.toml                 Configuración del paquete local
README.md                      Descripción e instrucciones de uso
```

## Formato del CSV

La aplicación recibe **una fila por ciclo de 60 segundos** y **68 columnas**:
17 sensores × 4 estadísticos (promedio, desviación estándar poblacional, mínimo
y máximo). Las mediciones crudas se resumen antes de cargar el archivo.

- Los nombres deben coincidir con las columnas de
  [config.py](src/ic_g7/config.py), por ejemplo `PS1_mean`, `PS1_std`, `PS1_min`
  y `PS1_max`. La app ordena las columnas automáticamente.
- Codificación UTF-8, con o sin BOM; separador de comas y punto decimal.
- Se admite uno o varios ciclos, hasta **5 MB**. Cada fila produce un resultado.
- No incluir `accumulator`, `clase`, `cycle_id`, `group_id` ni `stable`.
  No subir `features.csv` completo, porque contiene etiquetas e identificadores.

En **Ciclo de ejemplo → Descargar ejemplo** se obtiene una plantilla con las
68 columnas y un ciclo de entrenamiento.

## Validaciones

El archivo se rechaza completo, sin predicción, si:

- Está vacío, no contiene ciclos o supera el límite de tamaño.
- Faltan columnas, sobran columnas o hay nombres duplicados.
- Alguna fila tiene una cantidad incorrecta de campos.
- Hay valores nulos, no numéricos o infinitos.
- Hay valores fuera del rango de entrenamiento, con una tolerancia mínima por
  redondeo numérico.

La app muestra el motivo del rechazo y, cuando corresponde, el registro y la
columna afectados. Los rangos se calculan exclusivamente con entrenamiento.

## Experimentos de Random Forest y MLP

El aporte de Dayron está en
[04_modelos_dayron.ipynb](notebooks/04_modelos_dayron.ipynb). Contiene el
procedimiento explicado, las salidas ejecutadas y las limitaciones de la comparación.
La lógica se reutiliza desde `src/ic_g7/experimentos_dayron.py`.

Con el entorno y los datos preparados, ejecutar **el notebook completo o el script**:

```powershell
& ".\.venv\Scripts\python.exe" scripts/entrenar_modelos_dayron.py
```

Se evalúan Random Forest por defecto, Random Forest ajustado y MLP por defecto,
usando solo los **1827 ciclos de entrenamiento**, la partición compartida y los
cinco pliegues por grupos. Todos usan `crear_pipeline` y `validar_cruzado`.
La MLP conserva los valores predeterminados salvo `max_iter=4000` y la semilla
de `config.py`, como acordó José.

La búsqueda de RF prueba 27 combinaciones: `n_estimators` en 100, 200 y 300;
`max_depth` en `None`, 10 y 20; `min_samples_leaf` en 1, 2 y 4.
Se selecciona por F1 macro. La ejecución es en serie, con un hilo BLAS;
el hardware, las versiones, el coste de búsqueda y los hashes de datos y
partición quedan en `reports/entorno_dayron.json`.

El experimento genera únicamente sus archivos en `reports/`:

| Archivo | Contenido |
|---|---|
| `resultados_dayron.csv` | Tres filas con las columnas de la tabla común. |
| `pliegues_dayron.csv` | F1 macro, exactitud y ROC-AUC de cada pliegue. |
| `metricas_clase_dayron.csv` | Precisión, recall, F1 y soporte de cada categoría. |
| `matrices_confusion_dayron.csv` | Recuentos reales frente a predichos, en bar. |
| `predicciones_oof_dayron.csv` | Predicción y probabilidades de cada ciclo excluido del ajuste de su pliegue. |
| `busqueda_rf_dayron.csv` | Las 27 configuraciones evaluadas y sus puntuaciones y tiempos. |
| `figuras/matrices_confusion_dayron.png` | Las tres matrices de confusión. |

Las OOF agrupan los ciclos de validación; sus métricas por clase no son el promedio
entre pliegues. `indice_entrenamiento` identifica la posición dentro de
`cargar_entrenamiento()`, no el ciclo original. Las probabilidades todavía no
están calibradas. La MLP base de Dayron reproduce la configuración de José con
semilla 42: **no debe contarse como otro método ni como una repetición independiente**.

La selección de hiperparámetros y la comparación reutilizan los pliegues; las
cifras del RF ajustado pueden tener sesgo de selección. Son resultados de
validación, no de prueba ni una selección definitiva. Los tiempos por modelo
no incluyen la búsqueda ni la segunda pasada OOF; el coste de búsqueda se informa
por separado. Las diferencias pequeñas no prueban superioridad ni equivalencia.

Esta ejecución no consulta prueba, no exporta el modelo final y no modifica
Streamlit ni la tabla consolidada. La evaluación única en prueba y la integración
del modelo y del umbral corresponden a las siguientes etapas del equipo.

## Tabla de resultados

Cada persona escribe sus modelos en `reports/resultados_<nombre>.csv` y no edita
`reports/resultados.csv`, que es la tabla única y se genera uniendo esos archivos.

Regenerarla desde la raíz del repositorio, cuando ya estén todos los
`resultados_<nombre>.csv` en la rama y antes de usar la tabla (comparación en el
informe o en la evaluación del 27/oct):

```powershell
& ".\.venv\Scripts\python.exe" scripts/consolidar_resultados.py
```

El script falla indicando archivo y fila si falta una columna, un nombre de modelo
está vacío o repetido, una métrica no tiene el formato `media ± std` o un tiempo no
es un número finito no negativo.

## Modelo elegido y evaluación final

Se utiliza **MLP+GA (semilla 42)**, con la configuración registrada en
`reports/resultados_jose.csv`. Desde la raíz del repositorio, con el entorno y
los datos preparados, reentrenar con los **1827 ciclos de entrenamiento**:

```powershell
& ".\.venv\Scripts\python.exe" scripts/entrenar_modelo_final.py
```

Genera en `models/`:

- `pipeline_final.joblib`: pipeline completo entrenado.
- `labels.json`: clases, presiones y estados.
- `metadata_final.json`: configuración, columnas y datos de reproducibilidad.

El script no utiliza prueba y rechaza sobrescribir artefactos existentes.

**La evaluación de los 378 ciclos de prueba se realiza una sola vez el martes
27/oct.** Según el acuerdo actualizado, no requiere reunión ni la presencia
simultánea de los tres integrantes. En esa fecha:

```powershell
& ".\.venv\Scripts\python.exe" scripts/evaluar_modelo_final.py --confirmar-evaluacion
```

Guarda las métricas, la matriz de confusión, las predicciones y el registro en
`reports/evaluacion_final/`. Bloquea nuevos intentos incluso si hubo un fallo;
en ese caso se revisa con el equipo sin borrar el registro.

Ejecutar las pruebas locales:

```powershell
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -v
```

## Cómo levantar Streamlit

Se requiere **Python 3.12**. Ejecutar los comandos desde la raíz del repositorio
en PowerShell. Si el entorno y los datos ya están preparados, pasar directamente
al comando de Streamlit.

Preparación inicial:

```powershell
py -3.12 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv\Scripts\python.exe" scripts/descargar_datos.py
& ".\.venv\Scripts\python.exe" scripts/construir_features.py
```

Iniciar la aplicación:

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run app/app.py --server.address 127.0.0.1
```

Abrir http://127.0.0.1:8501. Mantener abierta la terminal; `Ctrl+C` detiene el
servidor. Cargar un CSV o el ciclo de ejemplo y pulsar **Analizar datos**.

**Esta versión usa Dummy provisionalmente:** devuelve siempre 90 bar para probar
la interfaz. Sus respuestas y probabilidades no son un diagnóstico; el modelo
definitivo se integrará en una etapa posterior.
