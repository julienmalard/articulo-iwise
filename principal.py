import os.path
from os import path, makedirs

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from constantes import COL_REGIÓN, COLS_REGIONES, DIR_EGRESO, COL_PAÍS, COL_SEGGHÍD, COL_SEGHÍD_BRUTA, COLS_PREGUNTAS, \
    COL_PESOS, EXPLORATORIO
from geografía import Geografía
from modelo import Modelo, ConfigDatos
from traducciones import guardar_traducciones


def preparar_datos():
    datos_pd = pd.read_csv("datos/IWISE2020-2022_LatinAmerica_20230615_Labels_utf8.csv")

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
    args_shp={"encoding": "latin-1"},
    traslado_nombres={"Belice": None}  # Sin comentario
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
    })

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
    modelo = Modelo("Región", var_y=COL_SEGGHÍD, var_x=COL_REGIÓN, config=config)

    if EXPLORATORIO:
        modelo.dibujar()
        Brazil.dibujar(modelo, colores=-1, escala_común=True)
        Guatemala.dibujar(modelo, colores=-1, escala_común=True)
        Honduras.dibujar(modelo, colores=-1, escala_común=True)
        Perú.dibujar(modelo, colores=-1, escala_común=True)

        Modelo("Género", var_y=COL_SEGGHÍD, var_x="WP1219", config=config).dibujar()

        Modelo("Ruralidad", var_y=COL_SEGGHÍD, var_x="degurba", config=config).dibujar()

        Modelo("Matrimonio", var_y=COL_SEGGHÍD, var_x="WP1223", config=config).dibujar()

        Modelo("Nivel educativo", var_y=COL_SEGGHÍD, var_x="WP3117", config=config).dibujar()

        Modelo("Empleo", var_y=COL_SEGGHÍD, var_x="EMP_2010", config=config).dibujar()

        Modelo("Religión", var_y=COL_SEGGHÍD, var_x="WP1233RECODED", config=config).dibujar()

        Modelo("Dificultad económica", var_y=COL_SEGGHÍD, var_x="WP2319", config=config).dibujar()

        Modelo("Clase económica", var_y=COL_SEGGHÍD, var_x="INCOME_5", config=config).dibujar()

    else:
        dir_figuras = os.path.join(config.dir_egreso, 'publicación')
        if not path.isdir(dir_figuras):
            makedirs(dir_figuras)

        # Figura 1 - Mapas
        fig, ejes = plt.subplots(2, 2, figsize=(16, 12))
        fig.subplots_adjust(top=0.85, bottom=0, wspace=0.01, hspace=0.05)
        for eje in [e for x in ejes for e in x]:
            eje.set_aspect('equal', 'box')
            eje.axis('off')

        Brazil.dibujar(modelo, colores=-1, escala_común=True, eje=ejes[0][0])
        Guatemala.dibujar(modelo, colores=-1, escala_común=True, eje=ejes[1][0])
        Honduras.dibujar(modelo, colores=-1, escala_común=True, eje=ejes[0][1])
        resultados = Perú.dibujar(modelo, colores=-1, escala_común=True, eje=ejes[1][1])

        ejes[0][0].set_title('Brazil', fontsize=18)
        ejes[1][0].set_title('Guatemala', fontsize=18)
        ejes[0][1].set_title('Honduras', fontsize=18)
        ejes[1][1].set_title('Perú', fontsize=18)

        fig.colorbar(resultados['colores'], ax=ejes[:, 1], location='right', shrink=0.6)

        fig.suptitle('Probabilidad de inseguridad hídrica', fontsize=35)
        fig.savefig(os.path.join(dir_figuras, 'Figura 1'))

        # Figura 3 - Género
        def generar_ejes():
            fig, ejes = plt.subplots(2, 2, figsize=(16, 12))
            fig.subplots_adjust(top=0.9, bottom=0, wspace=0.01, hspace=0.01)
            for eje in [e for x in ejes for e in x]:
                eje.set_aspect('equal', 'box')
                eje.axis('off')
            return fig, ejes

        modelo_género = Modelo("Género", var_y=COL_SEGGHÍD, var_x="WP1219", config=config)

        fig, ejes = generar_ejes()

        modelo_género.dibujar_caja_bigotes(ejes=ejes)
        fig.suptitle('Probabilidad de inseguridad hídrica por género', fontsize=35)
        fig.savefig(os.path.join(dir_figuras, 'Figura 3'))

        # Figura 4 - Urbanismo

        # Figura 5a - Ingresos

        # Figura 5b Dificultad económica

    guardar_traducciones()
