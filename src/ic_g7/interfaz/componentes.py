"""Piezas de presentación reutilizables para la interfaz del acumulador."""
import base64
from html import escape

import pandas as pd
import streamlit as st

from ic_g7 import config

TRAZOS = {
    "gota": '<path d="M12 3C9 7 5 11 5 15a7 7 0 0 0 14 0c0-4-4-8-7-12Z"/><path d="M8 15a4 4 0 0 0 3 4"/>',
    "sensor": '<circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="2"/><path d="M3 9V5a2 2 0 0 1 2-2h4m6 0h4a2 2 0 0 1 2 2v4M3 15v4a2 2 0 0 0 2 2h4m6 0h4a2 2 0 0 0 2-2v-4"/>',
    "tabla": '<rect x="3" y="4" width="18" height="17" rx="2"/><path d="M3 10h18M9 10v11m6-11v11M3 15h18"/>',
    "acumulador": '<rect x="7" y="7" width="10" height="15" rx="4"/><path d="M10 7V4a2 2 0 0 1 4 0v3M7 11h10"/>',
    "archivo": '<path d="M14 3H6a2 2 0 0 0-2 2v15a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9Z"/><path d="M14 3v6h6"/>',
    "subir": '<path d="M12 16V3m-5 5 5-5 5 5M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5"/>',
    "revision": '<path d="M4 20V10m6 10V4m6 16v-7m5 7H2"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="m7 12 3 3 7-7"/>',
}


def icono(nombre, clase=""):
    """Icono SVG local como imagen; compatible con el saneado HTML de Streamlit."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        'fill="none" stroke="#75b8ff" stroke-width="1.7" '
        'stroke-linecap="round" stroke-linejoin="round">'
        + TRAZOS[nombre] + "</svg>"
    )
    contenido = base64.b64encode(svg.encode()).decode()
    return (
        f'<img class="ui-icon {escape(clase)}" alt="" aria-hidden="true" '
        f'src="data:image/svg+xml;base64,{contenido}">'
    )


def cabecera():
    *primeras, ultima = config.CLASE_POR_OBJETIVO
    st.html(
        '<div class="ui-encabezado"><div><h1>Estado del acumulador hidráulico</h1>'
        '<p>Carga un CSV de sensores para evaluar el estado del acumulador.</p>'
        '</div><span class="ui-badge ui-badge-prueba">MLP + GA</span></div>'
        '<div class="ui-estadisticas">'
        + "".join(
            f'<div class="ui-estadistica"><span class="ui-icon-box">{icono(ico)}</span>'
            f'<div><strong>{valor}</strong><span>{texto}</span></div></div>'
            for ico, valor, texto in [
                ("sensor", len(config.SENSORES), "sensores"),
                ("tabla", len(config.COLUMNAS_ENTRADA), "características"),
                ("acumulador", len(config.CLASE_POR_OBJETIVO),
                 f"categorías: {', '.join(map(str, primeras))} y {ultima} bar"),
            ]
        )
        + "</div>"
    )


def titulo_panel(texto, nombre_icono, ancla=None, detalle=None):
    identificador = f' id="{escape(ancla)}"' if ancla else ""
    etiqueta = f'<span class="ui-badge">{escape(detalle)}</span>' if detalle else ""
    st.html(
        f'<div class="ui-titulo-panel"{identificador}>'
        f'{icono(nombre_icono)}<h2>{escape(texto)}</h2>{etiqueta}</div>'
    )


def barra_lateral(paso):
    with st.sidebar:
        st.html(
            '<div class="ui-marca"><span class="ui-logo">'
            + icono("gota") + '</span><strong>Acumulador<br>Hidráulico</strong></div>'
            '<nav class="ui-pasos" aria-label="Etapas del análisis">'
            + "".join(
                f'<a class="ui-paso {"activo" if numero == paso else ""}" '
                f'href="#{ancla}" target="_self" '
                + ('aria-current="step" ' if numero == paso else "")
                + f'><span class="ui-numero">{numero}</span>{icono(ico)}'
                f'<span>{nombre}</span></a>'
                for numero, nombre, ico, ancla in [
                    (1, "Datos", "archivo", "datos"),
                    (2, "Revisión", "revision", "revision"),
                    (3, "Respuesta", "acumulador", "respuesta"),
                ]
            )
            + '</nav><div class="ui-sidebar-nota"><span class="ui-punto"></span>'
            'Aplicación local · G7</div>'
        )



def entrada_ejemplo(cargado):
    titulo = "Ciclo de ejemplo listo" if cargado else "Prueba con un ciclo de ejemplo"
    texto = (
        "Puedes revisar la entrada y continuar con el análisis."
        if cargado else "Carga un registro de entrenamiento con las 68 características."
    )
    st.html(
        '<div class="ui-ejemplo">' + icono("archivo")
        + f"<strong>{titulo}</strong><span>{texto}</span></div>"
    )


def vista_previa(datos):
    titulo_panel(
        "Vista previa de la entrada", "tabla", ancla="revision",
        detalle="4 de 68 características" if datos is not None else "Sin datos",
    )
    if datos is None:
        st.html(
            '<div class="ui-vacio">' + icono("tabla")
            + "<strong>Tus datos aparecerán aquí</strong>"
            "<span>Sube un CSV o carga el ciclo de ejemplo.</span></div>"
        )
        return
    st.dataframe(datos.iloc[:4, :4], hide_index=True, width="stretch")
    st.caption(f"{len(datos)} ciclo(s) cargado(s) · Una fila por ciclo de 60 segundos.")
    with st.expander("Ver las 68 características"):
        st.dataframe(datos.head(10), hide_index=True, width="stretch")
        if len(datos) > 10:
            st.caption("Se muestran los primeros 10 ciclos.")


def validacion(datos, error):
    titulo_panel("Validación de datos", "check")
    if error:
        st.html('<div class="ui-estado-error">Revisa el archivo para continuar.</div>')
        st.error(error)
        return
    correcta = datos is not None
    textos = (
        ["Archivo CSV válido", "68 características detectadas",
         "Formato de columnas correcto", "Sin valores faltantes",
         "Valores numéricos dentro de rango"]
        if correcta else
        ["Archivo CSV pendiente", "Se esperan 68 características",
         "La validación se realiza al cargar los datos"]
    )
    clase = "correcta" if correcta else "pendiente"
    st.html(
        '<ul class="ui-validacion">'
        + "".join(
            f'<li class="{clase}"><span class="ui-check">'
            + ("✓" if correcta else "·") + f"</span>{texto}</li>"
            for texto in textos
        )
        + "</ul>"
    )


def respuesta(resultado):
    """Muestra la estimación del modelo final por ciclo y permite descargarla."""
    titulo_panel("Respuesta del acumulador", "acumulador", detalle="MLP + GA")
    if len(resultado) == 1:
        fila = resultado.iloc[0]
        nombres = {
            90: "Cerca de falla total", 100: "Presión muy reducida",
            115: "Presión algo reducida", 130: "Presión óptima",
        }
        izquierda, derecha = st.columns([1, 1.25])
        with izquierda:
            bar = int(fila["Acumulador (bar)"])
            st.html(
                '<div class="ui-resultado"><span>Categoría estimada</span>'
                f'<strong>{bar}<small> bar</small></strong>'
                f'<p>{nombres[bar]}</p><span class="ui-badge">'
                "Modelo final</span></div>"
            )
        with derecha:
            probabilidades = pd.DataFrame({
                "Categoría": [f"{bar} bar" for bar in config.CLASE_POR_OBJETIVO],
                "Probabilidad": [
                    f'{fila[f"{bar} bar (%)"]:.1f} %'
                    for bar in config.CLASE_POR_OBJETIVO
                ],
            })
            st.dataframe(probabilidades, hide_index=True, width="stretch")
    else:
        st.write(f"Resultados para {len(resultado)} ciclos.")
        st.dataframe(resultado, hide_index=True, width="stretch")

    st.caption(
        "Estimación del modelo MLP + GA. Las probabilidades no garantizan certeza; "
        "el resultado apoya la revisión del acumulador."
    )
    if resultado["Baja confianza"].any():
        st.warning("Hay registros cuya probabilidad máxima está bajo el umbral.")
    st.download_button(
        "Descargar resultados",
        data=resultado.to_csv(index=False).encode("utf-8"),
        file_name="resultados_acumulador.csv",
        mime="text/csv", on_click="ignore",
        icon=":material/download:",
    )
