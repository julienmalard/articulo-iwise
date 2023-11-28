import math
import os.path
from os import path, makedirs
from typing import Optional
from typing import TYPE_CHECKING

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import scipy as sp
import seaborn as sns

from constantes import VALS_EXCLUIR, IDIOMA
from traducciones import traducir

if TYPE_CHECKING:
    from geografía import Geografía


def prob(x, n_niveles, dv, niv, c, prg):
    x_transf = x * c.loc[{"c_dim_1": prg}].values
    if niv == 0:
        return 1 - sp.special.expit(x_transf - dv.loc[{"divisiones_dim_1": 0}].values)
    elif niv < n_niveles - 1:
        return sp.special.expit(x_transf - dv.loc[{"divisiones_dim_1": niv - 1}].values) - sp.special.expit(
            x_transf - dv.loc[{"divisiones_dim_1": niv}].values)
    else:
        return sp.special.expit(x_transf - dv.loc[{"divisiones_dim_1": niv - 1}].values)


def prob_cum(x, n_niveles, dv, niv, c, prg):
    probs = []
    for n in list(range(0, niv + 1)):
        probs.append(prob(x, n_niveles, dv, n, c, prg))
    return np.array(probs).sum(axis=0)


class ConfigDatos(object):
    def __init__(
            símismo, datos: pd.DataFrame,
            dir_egreso: str,
            col_país: str,
            col_región: str,
            cols_preguntas: Optional[str] = None,
            col_pesos: Optional[str] = None
    ):
        símismo.datos = datos
        símismo.dir_egreso = dir_egreso
        símismo.col_país = col_país
        símismo.col_región = col_región
        símismo.cols_preguntas = cols_preguntas
        símismo.col_pesos = col_pesos


class Modelo(object):
    def __init__(símismo, nombre: str, var_y: str, var_x: str, config: ConfigDatos):
        símismo.nombre = nombre
        símismo.var_y = var_y
        símismo.var_x = var_x
        símismo.config = config
        símismo.datos = config.datos[
            list({
                var_y, var_x, config.col_país, config.col_región,
                *(config.cols_preguntas or []),
                *([config.col_pesos] if config.col_pesos else [])
            })
        ].dropna()
        símismo.países = símismo.obt_datos()[config.col_país].unique()

        símismo.recalibrado = False

    def calibrar(símismo, país: str):
        datos_país = símismo.obt_datos(país)
        x_categórico = símismo.obt_x_categórico(país)

        with pm.Model():
            # Referencia: https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/
            mu_a = pm.Normal(name="mu_a", sigma=.1)
            sigma_a = pm.HalfNormal(name="sigma_a", sigma=.1)

            ajuste_a = pm.Normal(name="ajuste_a", mu=0, sigma=1, shape=x_categórico.categories.size)
            a = pm.Deterministic("a", mu_a + ajuste_a * sigma_a)

            índices_a = x_categórico.codes
            y = pm.ConstantData("y", datos_país[símismo.var_y])
            if símismo.config.col_pesos:
                # https://discourse.pymc.io/t/how-to-run-logistic-regression-with-weighted-samples/5689/8
                log_p = datos_país[símismo.config.col_pesos].values * pm.logp(
                    pm.Bernoulli.dist(logit_p=a[índices_a], name="prob"), y)
                pm.Potential("error", log_p)
            else:
                pm.Bernoulli(logit_p=a[índices_a], name="prob", observed=y)
            pm.Deterministic("b", pm.math.invlogit(a))

            traza = pm.sample()

        az.to_netcdf(traza, símismo.archivo_calibs(país))
        símismo.recalibrado = True

    def dibujar(símismo, recalibrar=False):
        símismo.dibujar_traza(recalibrar)
        símismo.dibujar_caja_bigotes(recalibrar)

        return símismo

    def dibujar_traza(símismo, recalibrar=False, ejes: Optional[plt.Axes] = None):
        países = símismo.datos[símismo.config.col_país].unique()

        for país in países:
            traza = símismo.obt_traza(país, recalibrar)

            categorías_x = símismo.obt_categorías_x(país)

            az.plot_trace(traza, ["b", "a", "mu_a", "sigma_a"], axes=ejes)
            if not ejes:
                fig = plt.gcf()
                fig.suptitle(f"{país.país}: Probabilidad por {', '.join(categorías_x.tolist())}")
                fig.savefig(símismo.archivo_gráfico(país, "traza"))
                plt.close(fig)

    def dibujar_caja_bigotes(símismo, recalibrar=False, ejes: Optional[list[plt.Axes]] = None):
        países = símismo.datos[símismo.config.col_país].unique()

        for país in países:
            fig = None
            if ejes is None:
                fig, ejes = plt.subplots(1, 2, figsize=(12, 6))
                fig.subplots_adjust(bottom=0.25)

            dibujo_dist, traza_por_categoría, categorías_x = símismo.dibujar_histograma(
                país=país, eje=ejes[0], recalibrar=recalibrar
            )
            n_categ = len(traza_por_categoría.columns)

            # Ajustar leyenda
            sns.move_legend(
                dibujo_dist, ncols=max(2, math.ceil(n_categ / 3)), bbox_to_anchor=(1.1, -0.23),
                loc="center"
            )
            colores = {
                categorías_x[i]: dibujo_dist.legend_.legendHandles[i].get_color() for i in range(n_categ)
            }
            símismo.dibujar_caja(país=país, eje=ejes[1], colores_por_categ=colores, recalibrar=recalibrar)

            if fig:
                fig.suptitle(f"{país}: Probabilidad de inseguridad hídrica por {símismo.nombre.lower()}")
                fig.savefig(símismo.archivo_gráfico(país, "caja"))
                plt.close(fig)

    def dibujar_histograma(símismo, país: str, eje: plt.Axes, leyenda=True, recalibrar=False):
        traza_por_categoría = símismo.obt_traza_por_categoría(país, recalibrar)
        traza_por_categoría = traza_por_categoría.rename({
            c: traducir(c, IDIOMA) for c in traza_por_categoría
        })
        categorías_x = símismo.obt_categorías_x(país).values.tolist()

        # Dibujar distribución
        dibujo_dist = sns.kdeplot(traza_por_categoría, ax=eje, legend=leyenda)

        eje.set_xlabel("Probabilidad de inseguridad hídrica", fontdict={"size": 14})
        eje.set_ylabel("Densidad", fontdict={"size": 14})

        return dibujo_dist, traza_por_categoría, categorías_x

    def dibujar_caja(símismo, país: str, eje: plt.Axes, colores_por_categ: dict, recalibrar=False):
        categorías_x = símismo.obt_categorías_x(país).values.tolist()
        traza_por_categoría = símismo.obt_traza_por_categoría(país, recalibrar)
        traza_por_categoría = traza_por_categoría.rename({
            c: traducir(c, IDIOMA) for c in traza_por_categoría
        })

        # Dibujar caja
        caja = traza_por_categoría.boxplot(ax=eje, grid=False, return_type="dict")

        eje.set(xticklabels=[])
        eje.set_xlabel(símismo.nombre)
        eje.set_ylabel("Probabilidad de inseguridad hídrica")

        for categ, color in colores_por_categ.items():
            i = categorías_x.index(categ)
            for forma in ["boxes", "medians"]:
                caja[forma][i].set_color(color)
            caja["fliers"][i].set_markeredgecolor(color)
            for forma in ["whiskers", "caps"]:
                for j in range(2):
                    caja[forma][i * 2 + j].set_color(color)

    def obt_traza(símismo, país: str, recalibrar=False):
        if (recalibrar and not símismo.recalibrado) or not path.isfile(símismo.archivo_calibs(país)):
            símismo.calibrar(país)
        return az.from_netcdf(símismo.archivo_calibs(país))

    def obt_datos(símismo, país: Optional[str] = None):
        datos = símismo.datos
        if país:
            datos = datos.loc[símismo.datos[símismo.config.col_país] == país]

        for v in VALS_EXCLUIR:
            datos = datos.loc[datos[símismo.var_x] != v]

        if datos[símismo.var_x].dtype == "category":
            datos[símismo.var_x] = datos[símismo.var_x].cat.remove_unused_categories()

        datos = datos.dropna()

        return datos

    def obt_x_categórico(símismo, país: str):
        datos_país = símismo.obt_datos(país)
        datos_x = datos_país[símismo.var_x]
        for c in datos_x.unique():
            datos_x = datos_x.replace(c, traducir(c, IDIOMA))
        return pd.Categorical(datos_x)

    def obt_categorías_x(símismo, país: str):
        return pd.Categorical(símismo.obt_x_categórico(país)).categories

    def obt_traza_por_categoría(símismo, país: str, recalibrar=False) -> pd.DataFrame:
        traza = símismo.obt_traza(país, recalibrar)

        categorías = traza.posterior["b_dim_0"].values
        categorías_x_datos = símismo.obt_categorías_x(país)

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

    def dibujar_valid(símismo, paises: Optional[list["Geografía"]] = None, por_categoría=True):
        paises = paises or [None]
        for país in paises:
            if por_categoría:
                categorías = símismo.datos[símismo.var_x].unique()
            else:
                categorías = [None]
            for categ in categorías:
                símismo.dibujar_preguntas(país, categ)
                símismo.dibujar_niveles(país, categ)

    def dibujar_preguntas(
            símismo, país: Optional["Geografía"] = None, categoría: Optional[str] = None,
            cuantiles=(0.5, .95, .99)
    ):
        todos_los_datos = símismo.obt_datos()
        traza = símismo.obt_traza_valid_por_país_y_categ(país, categoría)

        x = np.arange(-3, 3, 0.01)
        n_preguntas = len(símismo.config.cols_preguntas)
        n_niveles = np.unique(todos_los_datos[símismo.config.cols_preguntas]).size

        def traza_cuantil(q, v):
            return traza.posterior[v].quantile(q, dim=["draw", "chain"])

        def traza_cuantil_divs(q, prg):
            return traza_cuantil(q, "divisiones").loc[{"divisiones_dim_0": prg}]

        fig, ejes = plt.subplots(math.ceil((n_preguntas - 1) / 2), 2,
                                 figsize=(10, math.ceil((n_preguntas - 1) / 2) * 3))
        for pregunta in range(n_preguntas):
            eje = ejes[pregunta // 2, pregunta % 2]

            for nivel in range(n_niveles):
                y = prob(x, n_niveles, traza_cuantil_divs(0.5, pregunta), nivel, traza_cuantil(0.5, "c"), pregunta)
                ligne = eje.plot(x, y, label=f"nivel {nivel}")
                couleur = ligne[0].get_color()

                for q in cuantiles:
                    q_min = (1 - q) / 2
                    q_max = (1 - q) / 2 + 0.5
                    eje.fill_between(
                        x, y, prob(x, n_niveles, traza_cuantil_divs(q_min, pregunta), nivel, traza_cuantil(q_min, "c"),
                                   pregunta), alpha=0.1,
                        color=couleur
                    )
                    eje.fill_between(
                        x, y, prob(x, n_niveles, traza_cuantil_divs(q_max, pregunta), nivel, traza_cuantil(q_max, "c"),
                                   pregunta), alpha=0.1,
                        color=couleur
                    )

            nombre_pregunta = símismo.config.cols_preguntas[pregunta]
            eje.set_title(nombre_pregunta)

        handles, labels = ejes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center')

        fig.suptitle(f"Preguntas {categoría}, {país.país}")
        fig.supxlabel("Severidad")
        fig.supylabel("Probabilidad de nivel")
        fig.savefig(símismo.archivo_gráfico_valid(país, categoría, "preguntas"))
        plt.close(fig)

    def archivo_gráfico_valid(símismo, país, categoría, tipo):
        dir_gráfico = path.join(símismo.config.dir_egreso, "valid")
        if not path.isdir(dir_gráfico):
            makedirs(dir_gráfico)
        nombre_archivo = tipo
        if país:
            nombre_archivo += f"-{país.país}"
        if categoría:
            nombre_archivo += f"-{categoría}"
        return os.path.join(dir_gráfico, f"{nombre_archivo}.jpg")

    def dibujar_niveles(
            símismo, país: Optional["Geografía"] = None, categoría: Optional[str] = None, cuantiles=(0.5, .95, .99)
    ):
        todos_los_datos = símismo.obt_datos()
        traza = símismo.obt_traza_valid_por_país_y_categ(país, categoría)

        x = np.arange(-3, 3, 0.01)
        n_preguntas = len(símismo.config.cols_preguntas)
        n_niveles = np.unique(todos_los_datos[símismo.config.cols_preguntas]).size

        def traza_cuantil(q, v):
            return traza.posterior[v].quantile(q, dim=["draw", "chain"])

        def traza_cuantil_divs(q, prg):
            return traza_cuantil(q, "divisiones").loc[{"divisiones_dim_0": prg}]

        fig, ejes = plt.subplots(math.ceil((n_niveles - 1) / 2), 2, figsize=(10, math.ceil((n_niveles - 1) / 2) * 3))
        for nivel in range(n_niveles - 1):
            eje = ejes[nivel % 2, nivel // 2]

            for pregunta in range(n_preguntas):
                y = prob_cum(
                    x, n_niveles, traza_cuantil_divs(0.5, pregunta), nivel, traza_cuantil(0.5, "c"), pregunta
                )

                nombre_pregunta = símismo.config.cols_preguntas[pregunta]
                ligne = eje.plot(x, y, label=nombre_pregunta)
                couleur = ligne[0].get_color()

                for q in cuantiles:
                    q_min = (1 - q) / 2
                    q_max = (1 - q) / 2 + 0.5

                    eje.fill_between(
                        x, y, prob_cum(
                            x, n_niveles, traza_cuantil_divs(q_min, pregunta), nivel,
                            traza_cuantil(q_min, "c"), pregunta
                        ), alpha=0.1, color=couleur
                    )
                    eje.fill_between(
                        x, y, prob_cum(
                            x, n_niveles, traza_cuantil_divs(q_max, pregunta), nivel,
                            traza_cuantil(q_max, "c"), pregunta
                        ), alpha=0.1, color=couleur
                    )

            eje.set_title(f"Nivel inferior o igual a {nivel}")

        handles, labels = ejes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center')
        fig.supxlabel("Severidad")
        fig.supylabel("Probabilidad de nivel")
        fig.suptitle(f"Niveles {categoría}, {país.país}")

        fig.savefig(símismo.archivo_gráfico_valid(país, categoría, "niveles"))
        plt.close(fig)

    def validar(símismo, país: Optional["Geografía"] = None, categoría: Optional[str] = None):
        if not símismo.config.cols_preguntas:
            raise ValueError("Columnas de preguntas individuales no especificadas en la configuración del modelo.")
        datos = símismo.obt_datos()
        if país:
            datos = datos.loc[datos[símismo.config.col_país] == país.país]
        if categoría:
            datos = datos.loc[datos[símismo.var_x] == categoría]

        n_datos = len(datos)

        n_preguntas = len(símismo.config.cols_preguntas)
        n_niveles = np.unique(datos[símismo.config.cols_preguntas]).size

        with pm.Model():
            y = pm.ConstantData(
                "y", datos[símismo.config.cols_preguntas[0:n_preguntas]].values.astype(int).reshape(
                    (n_datos, n_preguntas)
                )
            )

            b = pm.Normal('b', mu=0, sigma=1, shape=n_datos)
            # c = pm.HalfNormal('c', sigma=1, shape=(1, n_preguntas - 1))
            # d = pm.math.concatenate([pm.math.ones(shape=(1, 1)), c], axis=1)
            d = pm.HalfNormal('c', sigma=1, shape=(1, n_preguntas))

            mu_divisiones = np.tile(
                np.arange(-1, n_niveles - 2),
                (n_preguntas, 1)
            )  # list(range(-1, n_niveles - 2))#

            divisiones = pm.Normal(
                name='divisiones', mu=mu_divisiones, sigma=10, shape=mu_divisiones.shape,  # n_niveles -1,#
                transform=pm.distributions.transforms.ordered
            )
            pm.OrderedLogistic(
                name="iwise", cutpoints=divisiones, eta=(d.T * b).T, observed=y, compute_p=False
            )

            traza = pm.sample()
        archivo = símismo.archivo_calibs_valid(país, categoría)
        az.to_netcdf(traza, archivo)

        az.plot_trace(traza, var_names=["divisiones", "c"])
        fig = plt.gcf()
        fig.suptitle("Traza")
        fig.savefig(símismo.archivo_gráfico(f"{categoría}-{país.país}", os.path.join("traza", "valid")))
        plt.close(fig)

    def obt_traza_valid_por_país_y_categ(
            símismo, país: Optional[str] = None, categoría: Optional[str] = None
    ):
        archivo = símismo.archivo_calibs_valid(país, categoría)
        if not path.isfile(archivo):
            símismo.validar(país, categoría)
        return az.from_netcdf(archivo)

    def archivo_calibs_valid(símismo, país: Optional["Geografía"] = None, categoría: Optional[str] = None) -> str:
        dir_calibs = path.join(símismo.config.dir_egreso, "calibs", "valid")
        if not path.isdir(dir_calibs):
            makedirs(dir_calibs)
        nombre_archivo = símismo.nombre
        if país:
            nombre_archivo += f"-{país.país}"
        if categoría:
            nombre_archivo += f"-{categoría}"
        return path.join(dir_calibs, f"{nombre_archivo}.ncdf")
