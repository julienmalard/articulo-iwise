import json
from typing import Optional

with open("traducciones.json", encoding="utf8") as d:
    traducciones = json.load(d)


def traducir(texto: str, idioma: str) -> Optional[str]:
    if texto not in traducciones:
        traducciones[texto] = {}
        return texto
    if idioma not in traducciones[texto]:
        return texto
    return traducciones[texto][idioma]


def guardar_traducciones():
    with open("traducciones.json", encoding="utf8", mode='w') as d:
        json.dump(traducciones, d, indent=2, ensure_ascii=False)
