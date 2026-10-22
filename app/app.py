"""Control del flujo de la app. Los componentes y estilos están separados."""
import hashlib

import streamlit as st

from ic_g7 import config
from ic_g7.interfaz import componentes as ui
from ic_g7.interfaz.tema import aplicar_tema
from ic_g7.prototipo import (
    predecir_provisional,
    preparar_prototipo,
    validar_csv,
)

st.set_page_config(
    page_title="Estado del acumulador",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="auto",
)


@st.cache_resource(show_spinner="Preparando la aplicación...")
def cargar_recursos(firma_features, firma_split):
    """Reutiliza entrenamiento; invalida la caché si cambian los archivos."""
    return preparar_prototipo()


def limpiar_respuesta():
    """Una entrada nueva no debe mostrar la respuesta de la entrada anterior."""
    st.session_state["resultado"] = None


def activar_ejemplo():
    # El callback se ejecuta antes del siguiente render de los controles.
    st.session_state["origen"] = "Ciclo de ejemplo"
    st.session_state["ejemplo_cargado"] = True
    limpiar_respuesta()


aplicar_tema()
ui.cabecera()

try:
    modelo, entrenamiento, limites = cargar_recursos(
        config.RUTA_FEATURES.stat().st_mtime_ns,
        config.RUTA_SPLIT.stat().st_mtime_ns,
    )
except (OSError, ValueError) as exc:
    ui.barra_lateral(1)
    st.error(f"No se pudieron preparar los datos de entrenamiento: {exc}")
    st.code(
        "python scripts/descargar_datos.py\n"
        "python scripts/construir_features.py", language="text",
    )
    st.stop()

ejemplo_csv = entrenamiento.head(1).to_csv(index=False).encode("utf-8")
contenido = None
st.session_state.setdefault("origen", "Subir CSV")
with st.container(key="panel_carga"):
    ui.titulo_panel("Cargar datos", "subir", ancla="datos")
    origen = st.radio(
        "Origen de los datos", ["Ciclo de ejemplo", "Subir CSV"],
        index=None, horizontal=True, label_visibility="collapsed",
        key="origen", on_change=limpiar_respuesta,
    )
    if origen == "Subir CSV":
        archivo = st.file_uploader(
            "Archivo CSV de sensores", type=["csv"], max_upload_size=5,
            label_visibility="collapsed", key="archivo_csv",
            on_change=limpiar_respuesta,
        )
        if archivo is not None:
            contenido = archivo.getvalue()
    else:
        cargado = st.session_state.get("ejemplo_cargado", False)
        ui.entrada_ejemplo(cargado)
        if cargado:
            contenido = ejemplo_csv

        acciones = st.columns([1.3, 1.3, 3], vertical_alignment="center")
        with acciones[0]:
            st.download_button(
                "Descargar ejemplo", data=ejemplo_csv,
                file_name="ejemplo_sensores.csv", mime="text/csv",
                on_click="ignore", icon=":material/download:", width="stretch",
            )
        with acciones[1]:
            st.button(
                "Cargar ejemplo", on_click=activar_ejemplo,
                icon=":material/description:", width="stretch",
            )
        with acciones[2]:
            st.html('<p class="ui-formato">Formato admitido: CSV · Máx. 5 MB</p>')


# El contenido y su origen identifican la entrada, incluso en los reruns de Streamlit.
firma = (origen, hashlib.sha256(contenido).hexdigest()) if contenido is not None else None
if firma != st.session_state.get("firma_entrada"):
    limpiar_respuesta()
    st.session_state["firma_entrada"] = firma

datos, error = None, None
if contenido is not None:
    try:
        datos = validar_csv(contenido, limites)
    except ValueError as exc:
        error = str(exc)

izquierda, derecha = st.columns([1.45, 1], gap="medium")
with izquierda:
    with st.container(key="panel_previa"):
        ui.vista_previa(datos)
with derecha:
    with st.container(key="panel_validacion"):
        ui.validacion(datos, error)
    analizar = st.button(
        "Analizar datos", type="primary", disabled=datos is None,
        icon=":material/monitoring:", width="stretch", key="analizar",
    )
    if analizar:
        st.session_state["resultado"] = predecir_provisional(modelo, datos)

st.html('<div id="respuesta" class="ui-ancla"></div>')
resultado = st.session_state.get("resultado")
if resultado is not None and datos is not None:
    with st.container(key="panel_respuesta"):
        ui.respuesta(resultado)

paso = 3 if resultado is not None and datos is not None else (2 if datos is not None else 1)
ui.barra_lateral(paso)
