import numpy as np

def convert_base_X(X):
    """
        Recibe el dataframe basico cargado que contiene ~1M de registros
        con dos columnas : "Text" y "Language" y las combina para
        crear un dataframe con solo registros de texto.

        Ademas, elimina palabras eliminables de los textos llamando a la funcion
        clean_text.
    """
    new_X = []
    for row in X.itertuples():
        print(f"\t{row.Index}/{len(X)}", end="\r")
        lan_tag = "lan_" + (row.Language if row.Language in language_map.keys() else "other")
        new_row = lan_tag + " " + row.Text
        new_X.append(new_row)
    return np.array(new_X)
