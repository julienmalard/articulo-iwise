import math
import os.path
from os import path, makedirs

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from constantes import COL_REGIÓN, COLS_REGIONES, DIR_EGRESO, COL_PAÍS, COL_SEGGHÍD, COL_SEGHÍD_BRUTA, COLS_PREGUNTAS, \
    COL_PESOS
from geografía import Geografía
from modelo import Modelo, ConfigDatos
from traducciones import guardar_traducciones


def preparar_datos():
    datos_pd = pd.read_csv("datos/IWISE2020-2022_LatinAmerica_20230615_Labels_utf8.csv")

    datos_pd.loc[datos_pd[COL_PAÍS] == "Peru", COL_PAÍS] = "Perú"

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
    país='Perú',
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

config = preparar_config()
modelo_geog = Modelo("Región", var_y=COL_SEGGHÍD, var_x=COL_REGIÓN, config=config)

modelo_género = Modelo("Género", var_y=COL_SEGGHÍD, var_x="WP1219", config=config)
modelo_ruralidad = Modelo("Ruralidad", var_y=COL_SEGGHÍD, var_x="degurba", config=config)
modelo_matrimonio = Modelo("Matrimonio", var_y=COL_SEGGHÍD, var_x="WP1223", config=config)
modelo_nivel_educativo = Modelo("Nivel educativo", var_y=COL_SEGGHÍD, var_x="WP3117", config=config)
modelo_empleo = Modelo("Empleo", var_y=COL_SEGGHÍD, var_x="EMP_2010", config=config)
modelo_religión = Modelo("Religión", var_y=COL_SEGGHÍD, var_x="WP1233RECODED", config=config)
modelo_dificultad_económica = Modelo("Dificultad económica", var_y=COL_SEGGHÍD, var_x="WP2319", config=config)
modelo_clase_económica = Modelo("Clase económica", var_y=COL_SEGGHÍD, var_x="INCOME_5", config=config)

if __name__ == "__main__":

    dir_figuras = os.path.join(config.dir_egreso, 'publicación')
    if not path.isdir(dir_figuras):
        makedirs(dir_figuras)


    def generar_figura_histograma(modelo: Modelo, cajas=False):
        fig, ejes = plt.subplots(2, 2, figsize=(16, 12))
        for i, país in enumerate([Brazil, Guatemala, Honduras, Perú]):
            eje = ejes[i % 2, i // 2]

            if cajas:
                dibujo, _, categorías_x = modelo.dibujar_histograma(país=país.país, eje=eje)
                colores = {
                    categorías_x[i]: dibujo.legend_.legend_handles[i].get_color() for i in range(len(categorías_x))
                }
                eje.clear()
                modelo.dibujar_caja(país=país.país, eje=eje, colores_por_categ=colores)
            else:
                dibujo, _, categs = modelo.dibujar_histograma(país=país.país, eje=eje, leyenda=i == 1)

                if i == 1:
                    sns.move_legend(
                        dibujo, ncols=max(2, math.ceil(len(categs) / 2)), bbox_to_anchor=(1.1, -0.23),
                        loc="center", prop={'size': 15}
                    )

            eje.set_title(país.país, fontsize=15)
        return fig, ejes


    # Figura 3 - Género
    fig, _ = generar_figura_histograma(modelo_género)

    fig.suptitle('Probabilidad de inseguridad hídrica por género', fontsize=35)
    fig.savefig(os.path.join(dir_figuras, 'Figura 3'))

    # Figura 4 - Urbanismo
    fig, _ = generar_figura_histograma(modelo_ruralidad)

    fig.suptitle('Probabilidad de inseguridad hídrica por ruralidad', fontsize=35)
    fig.savefig(os.path.join(dir_figuras, 'Figura 4'))

    # Figura 5a - Ingresos
    fig, _ = generar_figura_histograma(modelo_clase_económica)

    fig.suptitle('Probabilidad de inseguridad hídrica por ingresos', fontsize=35)
    fig.savefig(os.path.join(dir_figuras, 'Figura 5a'))

    # Figura 5b - Dificultad económica
    fig, _ = generar_figura_histograma(modelo_dificultad_económica)

    fig.suptitle('Probabilidad de inseguridad hídrica por dificultad económica', fontsize=35)
    fig.savefig(os.path.join(dir_figuras, 'Figura 5b'))

    # Figura 3 - Género
    fig, _ = generar_figura_histograma(modelo_género, cajas=True)

    fig.suptitle('Probabilidad de inseguridad hídrica por género', fontsize=35)
    fig.savefig(os.path.join(dir_figuras, 'Figura 3 cajas'))

    # Figura 4 - Urbanismo
    fig, _ = generar_figura_histograma(modelo_ruralidad, cajas=True)

    fig.suptitle('Probabilidad de inseguridad hídrica por ruralidad', fontsize=35)
    fig.savefig(os.path.join(dir_figuras, 'Figura 4 cajas'))

    # Figura 5a - Ingresos
    fig, _ = generar_figura_histograma(modelo_clase_económica, cajas=True)

    fig.suptitle('Probabilidad de inseguridad hídrica por ingresos', fontsize=35)
    fig.savefig(os.path.join(dir_figuras, 'Figura 5a cajas'))

    # Figura 5b - Dificultad económica
    fig, _ = generar_figura_histograma(modelo_dificultad_económica, cajas=True)

    fig.suptitle('Probabilidad de inseguridad hídrica por dificultad económica', fontsize=35)
    fig.savefig(os.path.join(dir_figuras, 'Figura 5b cajas'))

    # Figura 1 - Mapas
    fig, ejes = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(top=0.85, bottom=0, wspace=0.01, hspace=0.05)
    for eje in [e for x in ejes for e in x]:
        eje.set_aspect('equal', 'box')
        eje.axis('off')

    Brazil.dibujar(modelo_geog, colores=-1, escala_común=True, eje=ejes[0][0])
    Guatemala.dibujar(modelo_geog, colores=-1, escala_común=True, eje=ejes[1][0])
    Honduras.dibujar(modelo_geog, colores=-1, escala_común=True, eje=ejes[0][1])
    resultados = Perú.dibujar(modelo_geog, colores=-1, escala_común=True, eje=ejes[1][1])

    ejes[0][0].set_title('Brazil', fontsize=18)
    ejes[1][0].set_title('Guatemala', fontsize=18)
    ejes[0][1].set_title('Honduras', fontsize=18)
    ejes[1][1].set_title('Perú', fontsize=18)

    fig.colorbar(resultados['colores'], ax=ejes[:, 1], location='right', shrink=0.6)

    fig.suptitle('Probabilidad de inseguridad hídrica', fontsize=35)
    fig.savefig(os.path.join(dir_figuras, 'Figura 1'))

guardar_traducciones()
