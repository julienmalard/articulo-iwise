from principal import modelo_género, modelo_geog, Brazil, Guatemala, Honduras, Perú, modelo_ruralidad

paises = [Brazil, Guatemala, Honduras, Perú]

if __name__ == "__main__":
    modelo_género.dibujar_valid(paises)
    modelo_ruralidad.dibujar_valid(paises)

    modelo_género.dibujar_valid()
    modelo_ruralidad.dibujar_valid()

    modelo_geog.dibujar_valid(por_categoría=False)



    """
    modelo_geog.dibujar()
    Brazil.dibujar(modelo_geog, colores=-1, escala_común=True)
    Guatemala.dibujar(modelo_geog, colores=-1, escala_común=True)
    Honduras.dibujar(modelo_geog, colores=-1, escala_común=True)
    Perú.dibujar(modelo_geog, colores=-1, escala_común=True)

    modelo_género.dibujar()
    modelo_ruralidad.dibujar()
    modelo_matrimonio.dibujar()
    modelo_nivel_educativo.dibujar()
    modelo_empleo.dibujar()
    modelo_religión.dibujar()
    modelo_dificultad_económica.dibujar()
    modelo_clase_económica.dibujar()
    """
