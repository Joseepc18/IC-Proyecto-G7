"""Descarga y extrae el dataset Hydraulic Systems (UCI) en data/raw/hydraulic. Se puede repetir sin problema."""
import hashlib
import urllib.request
import zipfile

from ic_g7 import config


def sha256(ruta) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def main() -> None:
    config.DIR_CRUDO.mkdir(parents=True, exist_ok=True)
    ruta_zip = config.DIR_CRUDO / "hydraulic.zip"

    if not ruta_zip.exists():
        print(f"Descargando {config.URL_DATOS} (~77 MB)...")
        urllib.request.urlretrieve(config.URL_DATOS, ruta_zip)

    if sha256(ruta_zip) != config.SHA256_ZIP:
        raise SystemExit(
            "El SHA-256 del zip no coincide con el esperado: borra hydraulic.zip y vuelve a ejecutar. "
            "Si persiste, UCI cambio el archivo: avisa al equipo."
        )

    with zipfile.ZipFile(ruta_zip) as z:
        nombres = z.namelist()
        inesperados = set(nombres) - set(config.ARCHIVOS_ESPERADOS)
        if inesperados:
            raise SystemExit(f"El zip trae archivos inesperados: {sorted(inesperados)}")
        faltantes = [n for n in nombres if not (config.DIR_CRUDO / n).exists()]
        if faltantes:
            print(f"Extrayendo {len(faltantes)} archivos...")
            z.extractall(config.DIR_CRUDO, members=faltantes)

    print("Datos listos en", config.DIR_CRUDO)


if __name__ == "__main__":
    main()
