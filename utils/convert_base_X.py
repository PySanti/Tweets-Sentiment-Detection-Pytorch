import numpy as np
from .clean_text import clean_text

def convert_base_X(X, SUPPORTED_LAN=["en", "fr"]):
    """
        Recibe el dataframe basico cargado que contiene ~1M de registros
        con dos columnas : "Text" y "Language" y las combina para
        crear un dataframe con solo registros de texto.

        Ademas, elimina palabras eliminables de los textos llamando a la funcion
        clean_text.
    """
    def process_row(row):
        print(f"\t{row.name}/{len(X)}", end="\r")
        lan_tag = "lan_" + (row.Language if row.Language in SUPPORTED_LAN else "other")
        new_row = lan_tag + " " + clean_text(row.Text, row.Language)
        return new_row
    return X.apply(process_row, axis=1).values


