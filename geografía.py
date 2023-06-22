import warnings
from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import shapefile as sf
from matplotlib import colors, cm

from modelo import Modelo


class Geografía(object):
    def __init__(
            símismo, forma: Union[str, sf.Reader], país: str, columna_región: str, traslado_nombres=None,
            args_shp=None
    ):
        if isinstance(forma, str):
            símismo.forma = sf.Reader(forma, **(args_shp or {}))
        else:
            símismo.forma = forma
        símismo.país = país
        símismo.columna_región = columna_región
        símismo.traslado_nombres = traslado_nombres or {}

    def dibujar(
            símismo,
            modelo: Modelo,
            colores=None,
            llenar=True,
            escala_común=False,
            eje: Optional[plt.Axes] = None
    ):
        fig = None
        if not eje:
            fig, eje = plt.subplots(1, 1, figsize=(8, 6))
            eje.set_aspect('equal', 'box')

        traza = modelo.obt_traza_por_categoría(símismo.país)
        vals_por_región = traza.mean()
        desv_típ_por_región = traza.std()

        if escala_común:
            trazas = [modelo.obt_traza_por_categoría(país) for país in modelo.países]
            escala = (
                np.min(np.concatenate([t.mean() for t in trazas])), np.max(np.concatenate([t.mean() for t in trazas]))
            )
            escala_desv = (
                np.min(np.concatenate([t.std() for t in trazas])), np.max(np.concatenate([t.std() for t in trazas]))
            )
        else:
            escala = (min(vals_por_región), max(vals_por_región))
            escala_desv = (min(desv_típ_por_región), max(desv_típ_por_región))

        if escala[0] == escala[1]:
            escala = (escala[0] - 0.5, escala[0] + 0.5)

        vals_norm = (vals_por_región - escala[0]) / (escala[1] - escala[0])
        desv_típ_norm = (desv_típ_por_región - escala_desv[0]) / (
                escala_desv[1] - escala_desv[0]
        )

        escala_colores = símismo._resolver_colores(colores)
        d_clrs = _gen_d_mapacolores(colores=escala_colores)

        mapa_color = colors.LinearSegmentedColormap('mapa_color', d_clrs)
        norm = colors.Normalize(vmin=escala[0], vmax=escala[1])
        cpick = cm.ScalarMappable(norm=norm, cmap=mapa_color)
        cpick.set_array(np.array([]))

        v_cols = mapa_color(vals_norm)
        v_cols[np.isnan(vals_norm)] = 1

        def código(r):
            try:
                return símismo.traslado_nombres[r]
            except KeyError:
                return r

        regiones = [r[símismo.columna_región] for r in símismo.forma.records(fields=[símismo.columna_región])]
        faltan_en_mapa = [r for r in vals_norm.index.values.tolist() if r not in [código(s) for s in regiones]]
        if faltan_en_mapa:
            warnings.warn(f"Faltan las regiones siguientes en el mapa: {faltan_en_mapa}")

        for frm in símismo.forma.shapes():
            puntos = frm.points
            partes = frm.parts

            rgn = símismo.forma.record(frm.oid, fields=[símismo.columna_región])
            rgn_final = código(rgn[símismo.columna_región])

            if rgn_final is None:
                continue

            try:
                i_rgn = vals_norm.index.values.tolist().index(rgn_final)
                clr = v_cols[i_rgn]
                alpha = (1 - desv_típ_norm[i_rgn]) * 0.75 + 0.25
            except ValueError:
                warnings.warn(f"Región {rgn_final} no encontrada en los datos.")
                clr = [0.8, 0.8, 0.8]
                alpha = 1

            for ip, i0 in enumerate(partes):  # Para cada parte de la imagen

                if ip < len(partes) - 1:
                    i1 = partes[ip + 1] - 1
                else:
                    i1 = len(puntos)

                seg = puntos[i0:i1 + 1]
                x_lon = np.zeros((len(seg), 1))
                y_lat = np.zeros((len(seg), 1))
                for j in range(len(seg)):
                    x_lon[j] = seg[j][0]
                    y_lat[j] = seg[j][1]

                if llenar:
                    eje.fill(x_lon, y_lat, color=clr, alpha=alpha)
                else:
                    eje.plot(x_lon, y_lat, color=clr, alpha=alpha)

        if fig:
            fig.suptitle("Probabilidad de inseguridad hídrica")
            fig.colorbar(cpick, ax=eje)
            fig.savefig(modelo.archivo_gráfico(país=símismo.país, tipo="geog"))

        return {'colores': cpick}

    @staticmethod
    def _resolver_colores(colores=None):
        if colores is None:
            return ['#FF6666', '#FFCC66', '#00CC66']
        elif colores == -1:
            return ['#00CC66', '#FFCC66', '#FF6666']
        elif isinstance(colores, str):
            return ['#FFFFFF', colores]
        return colores


def _hex_a_rva(hx):
    """
    Convierte colores RVA a Hex.

    Parameters
    ----------
    hx: str
        El valor hex.

    Returns
    -------
    tuple
        El valor rva.
    """
    return tuple(int(hx.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))


def _gen_d_mapacolores(colores):
    """
    Genera un diccionario de mapa de color para MatPlotLib.

    Parameters
    ----------
    colores: list
        Una lista de colores

    Returns
    -------
    dict
        Un diccionario para MatPlotLib
    """

    clrs_rva = [_hex_a_rva(x) for x in colores]
    # noinspection PyTypeChecker
    n_colores = len(colores)

    dic_c = {'red': tuple((round(i / (n_colores - 1), 2), clrs_rva[i][0] / 255, clrs_rva[i][0] / 255) for i in
                          range(0, n_colores)),
             'green': tuple(
                 (round(i / (n_colores - 1), 2), clrs_rva[i][1] / 255, clrs_rva[i][1] / 255) for i in
                 range(0, n_colores)),
             'blue': tuple(
                 (round(i / (n_colores - 1), 2), clrs_rva[i][2] / 255, clrs_rva[i][2] / 255) for i in
                 range(0, n_colores))}

    return dic_c
