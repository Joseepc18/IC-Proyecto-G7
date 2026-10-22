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
tests/                        Comprobaciones del prototipo
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

## Tabla de resultados

Cada persona escribe sus modelos en `reports/resultados_<nombre>.csv` y no edita
`reports/resultados.csv`, que es la tabla única y se genera uniendo esos archivos.

Regenerarla desde la raíz del repositorio, cuando ya estén todos los
`resultados_<nombre>.csv` en la rama y antes de usar la tabla (comparación en el
informe o en la reunión del 27/oct):

```powershell
& ".\.venv\Scripts\python.exe" scripts/consolidar_resultados.py
```

El script falla indicando archivo y fila si falta una columna, un nombre de modelo
está vacío o repetido, una métrica no tiene el formato `media ± std` o un tiempo no
es un número finito no negativo.

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
