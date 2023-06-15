import math
from os import path, makedirs

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm
import seaborn as sns

from constantes import VALS_EXCLUIR, IDIOMA
from traducciones import traducir


class ConfigDatos(object):
    def __init__(
            símismo, datos: pd.DataFrame,
            dir_egreso: str,
            col_país: str,
            col_región: str
    ):
        símismo.datos = datos
        símismo.dir_egreso = dir_egreso
        símismo.col_país = col_país
        símismo.col_región = col_región


class Modelo(object):
    def __init__(símismo, nombre: str, var_y: str, var_x: str, config: ConfigDatos):
        símismo.nombre = nombre
        símismo.var_y = var_y
        símismo.var_x = var_x
        símismo.config = config
        símismo.datos = config.datos[list({var_y, var_x, config.col_país, config.col_región})].dropna()

        símismo.recalibrado = False

    def calibrar(símismo, país: str):
        datos_país = símismo.obt_datos_país(país)
        datos_x = datos_país[símismo.var_x]
        categorías_x = pd.Categorical(datos_x)

        with pm.Model():
            # Referencia: https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/
            mu_a = pm.Normal(name="mu_a", sigma=.1)
            sigma_a = pm.HalfNormal(name="sigma_a", sigma=.1)

            ajuste_a = pm.Normal(name="ajuste_a", mu=0, sigma=1, shape=categorías_x.categories.size)
            a = pm.Deterministic("a", mu_a + ajuste_a * sigma_a)

            índices_a = categorías_x.codes
            pm.Bernoulli(logit_p=a[índices_a], name="prob", observed=datos_país[símismo.var_y])

            pm.Deterministic("b", pm.math.invlogit(a))

            traza = pm.sample()

        az.to_netcdf(traza, símismo.archivo_calibs(país))
        símismo.recalibrado = True

    def dibujar(símismo, recalibrar=False):
        símismo.dibujar_traza(recalibrar)
        símismo.dibujar_caja_bigotes(recalibrar)

        return símismo

    def dibujar_traza(símismo, recalibrar=False):
        países = símismo.datos[símismo.config.col_país].unique()

        for país in países:
            traza = símismo.obt_traza(país, recalibrar)

            categorías_x = símismo.obt_categorías_x(país)
            símismo.obt_categorías_x(país)

            az.plot_trace(traza, ["b", "a", "mu_a", "sigma_a"])
            fig = plt.gcf()
            fig.suptitle(f"{país}: Probabilidad por {', '.join(categorías_x.categories.tolist())}")
            fig.savefig(símismo.archivo_gráfico(país, "traza"))
            plt.close(fig)

    def dibujar_caja_bigotes(símismo, recalibrar=False):
        países = símismo.datos[símismo.config.col_país].unique()

        for país in países:
            categorías_x = símismo.obt_categorías_x(país).categories.values.tolist()

            fig, ejes = plt.subplots(1, 2, figsize=(12, 6))
            fig.subplots_adjust(bottom=0.25)

            traza_por_categoría = símismo.obt_traza_por_categoría(país, recalibrar)
            traza_por_categoría = traza_por_categoría.rename({
                c: traducir(c, IDIOMA) for c in traza_por_categoría
            })
            n_categ = len(traza_por_categoría.columns)

            # Dibujar distribución
            dibujo_dist = sns.kdeplot(traza_por_categoría, ax=ejes[0])
            ejes[0].set_xlabel("Probabilidad de inseguridad hídrica")
            ejes[0].set_ylabel("Densidad")

            # Ajustar leyenda
            sns.move_legend(
                dibujo_dist, ncols=max(2, math.ceil(n_categ / 3)), bbox_to_anchor=(1.1, -0.23),
                loc="center"
            )

            # Dibujar caja
            caja = traza_por_categoría.boxplot(ax=ejes[1], grid=False, return_type="dict")
            colores_por_categ = {
                categorías_x[i]: dibujo_dist.legend_.legendHandles[i].get_color() for i in range(n_categ)
            }
            ejes[1].set(xticklabels=[])
            ejes[1].set_xlabel(símismo.nombre)
            ejes[1].set_ylabel("Probabilidad de inseguridad hídrica")

            for categ, color in colores_por_categ.items():
                i = categorías_x.index(categ)
                for forma in ["boxes", "medians"]:
                    caja[forma][i].set_color(color)
                caja["fliers"][i].set_markeredgecolor(color)
                for forma in ["whiskers", "caps"]:
                    for j in range(2):
                        caja[forma][i * 2 + j].set_color(color)

            fig.suptitle(f"{país}: Probabilidad de inseguridad hídrica por {símismo.nombre.lower()}")
            fig.savefig(símismo.archivo_gráfico(país, "caja"))
            plt.close(fig)

    def obt_traza(símismo, país: str, recalibrar=False):
        if (recalibrar and not símismo.recalibrado) or not path.isfile(símismo.archivo_calibs(país)):
            símismo.calibrar(país)
        return az.from_netcdf(símismo.archivo_calibs(país))

    def obt_datos_país(símismo, país: str):
        datos_país = símismo.datos.loc[símismo.datos[símismo.config.col_país] == país]

        if datos_país[símismo.var_x].dtype == "category":
            for v in VALS_EXCLUIR:
                datos_país = datos_país.loc[datos_país[símismo.var_x] != v]
            datos_país[símismo.var_x] = datos_país[símismo.var_x].cat.remove_unused_categories()

        datos_país = datos_país.dropna()

        return datos_país

    def obt_categorías_x(símismo, país: str):
        datos_país = símismo.obt_datos_país(país)
        datos_x = datos_país[símismo.var_x]
        for c in datos_x.unique():
            datos_x = datos_x.replace(c, traducir(c, IDIOMA))
        return pd.Categorical(datos_x)

    def obt_traza_por_categoría(símismo, país: str, recalibrar=False) -> pd.DataFrame:
        traza = símismo.obt_traza(país, recalibrar)

        categorías = traza.posterior["b_dim_0"].values
        categorías_x_datos = símismo.obt_categorías_x(país).categories

        return pd.DataFrame({
            categorías_x_datos[c]: traza.posterior["b"].sel({"b_dim_0": c}).values.flatten() for c in categorías
        })

    def archivo_calibs(símismo, país: str) -> str:
        dir_calibs = path.join(símismo.config.dir_egreso, "calibs")
        if not path.isdir(dir_calibs):
            makedirs(dir_calibs)
        return path.join(dir_calibs, f"{símismo.nombre}-{país}.ncdf")

    def archivo_gráfico(símismo, país: str, tipo: str) -> str:
        dir_gráfico = path.join(símismo.config.dir_egreso, tipo)
        if not path.isdir(dir_gráfico):
            makedirs(dir_gráfico)

        return path.join(dir_gráfico, f"{símismo.nombre}-{país}.jpg")
