import os.path

import numpy as np
import pandas as pd

from constantes import COL_REGIÓN, COLS_REGIONES, DIR_EGRESO, COL_PAÍS, COL_SEGGHÍD, COL_SEGHÍD_BRUTA, COLS_PREGUNTAS, \
    COL_PESOS
from geografía import Geografía
from modelo import Modelo, ConfigDatos
from traducciones import guardar_traducciones


def preparar_datos():
    datos_pd = pd.read_stata("datos/IWISE2020-2022_LatinAmerica.dta")

    # Combinar columnas de región
    datos_pd[COL_REGIÓN] = np.nan
    for col in COLS_REGIONES:
        datos_pd[COL_REGIÓN] = datos_pd[COL_REGIÓN].fillna(datos_pd[col])

    datos_pd[COL_SEGGHÍD] = np.where(datos_pd[COL_SEGHÍD_BRUTA] < 12, 0, 1)

    return datos_pd


def preparar_config():
    datos_pd = preparar_datos()
    return ConfigDatos(
        datos_pd, dir_egreso=DIR_EGRESO, col_país=COL_PAÍS, col_región=COL_REGIÓN, cols_preguntas=COLS_PREGUNTAS,
        col_pesos=COL_PESOS
    )


# Mapa del IARNA URL Guatemala
Guatemala = Geografía(
    os.path.join("geog", "mapas", "guatemala", "departamentos_gtm_fin"),
    país="Guatemala",
    columna_región="FIRST_DEPA",
    args_shp={"encoding": "latin-1"}
)

# https://data.humdata.org/dataset/cod-ab-hnd
Honduras = Geografía(
    os.path.join("geog", "mapas", "honduras", "hnd_admbnda_adm1_sinit_20161005"),
    país="Honduras",
    columna_región="ADM1_ES",
    traslado_nombres={
        "Atlantida": "Atlántida",
        "Cortes": "Cortés",
        "Islas de La Bahia": "Isla de la Bahía",
        "Santa Barbara": "Santa Bárbara",
        "Copan": "Copán",
        "Intibuca": "Intibucá",
        "Francisco Morazan": "Francisco Morazán",
        "El Paraiso": "El Paraíso"
    }
)

Brazil = Geografía(
    os.path.join("geog", "mapas", "brazil", "bra_admbnda_adm1_ibge_2020"),
    país="Brazil",
    columna_región="ADM1_PT",
    traslado_nombres={
        "Espírito Santo": "Espirito Santo",
        "Rondônia": "Rondonia"
    }
)

# https://data.humdata.org/dataset/cod-ab-per
Perú = Geografía(
    os.path.join("geog", "mapas", "perú", "per_admbnda_adm1_ign_20200714"),
    país='Peru',
    columna_región="ADM1_ES",
    traslado_nombres={
        "Amazonas": "amazonas",
        "Ancash": "ancash",
        "Apurimac": "apurimac",
        "Arequipa": "arequipa",
        "Ayacucho": "ayacucho",
        "Cajamarca": "cajamarca",
        # "Callao": ""
        "Cusco": "cusco",
        "Huancavelica": "huancavelica",
        "Huanuco": "huanuco",
        "Ica": "ica",
        "Junin": "junÍn",
        "La Libertad": "LA LIBERTAD",
        "Lambayeque": "lambayeque",
        "Lima": "lima",
        "Loreto": "loreto",
        "Madre de Dios": "MADRE DE DIOS",
        # "Moquegua": "",
        # "Pasco": "",
        "Piura": "piura",
        "Puno": "puno",
        "San Martin": "SAN MARTÍN",
        "Tacna": "tacna",
        "Tumbes": "tumbes",
        "Ucayali": "ucayali",
    }
)

if __name__ == "__main__":
    config = preparar_config()

    modelo = Modelo("Región", var_y=COL_SEGGHÍD, var_x=COL_REGIÓN, config=config).dibujar()

    Brazil.dibujar(modelo, colores=-1, escala_común=True)
    Guatemala.dibujar(modelo, colores=-1, escala_común=True)
    Honduras.dibujar(modelo, colores=-1, escala_común=True)
    Perú.dibujar(modelo, colores=-1, escala_común=True)

    Modelo("Género", var_y=COL_SEGGHÍD, var_x="WP1219", config=config).dibujar()

    Modelo("Ruralidad", var_y=COL_SEGGHÍD, var_x="WP14", config=config).dibujar()

    Modelo("Matrimonio", var_y=COL_SEGGHÍD, var_x="WP1223", config=config).dibujar()

    Modelo("Nivel educativo", var_y=COL_SEGGHÍD, var_x="WP3117", config=config).dibujar()

    Modelo("Empleo", var_y=COL_SEGGHÍD, var_x="EMP_2010", config=config).dibujar()

    Modelo("Religión", var_y=COL_SEGGHÍD, var_x="WP1233RECODED", config=config).dibujar()

    Modelo("Dificultad económica", var_y=COL_SEGGHÍD, var_x="WP2319", config=config).dibujar()

    Modelo("Clase económica", var_y=COL_SEGGHÍD, var_x="INCOME_5", config=config).dibujar()

    guardar_traducciones()
