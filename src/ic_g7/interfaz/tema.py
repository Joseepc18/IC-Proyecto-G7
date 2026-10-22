"""Carga de recursos visuales locales; no requiere servicios externos."""
import base64

import streamlit as st

from ic_g7 import config


def aplicar_tema():
    """Inserta los estilos y el fondo decorativo sin alterar los datos."""
    recursos = config.RAIZ / "app" / "assets"
    fondo = base64.b64encode((recursos / "hidraulica.svg").read_bytes()).decode()
    estilos = (recursos / "estilos.css").read_text(encoding="utf-8")
    estilos = estilos.replace("__FONDO_HIDRAULICO__", fondo)
    st.html("<style>" + estilos + "</style>")
