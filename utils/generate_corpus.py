from .constants import MIN_APP_FREC
import numpy as np
import pandas as pd
def generate_corpus(X):
    """
        Recibe la lista de tweets limpios y retorna el corpus.
    """
    app_dict = {}
    for tw in X:
        print(tw)
        for w in tw.split(" "):
            if app_dict.get(w):
                app_dict[w] += 1
            else:
                app_dict[w] = 1
    return np.array([w for w in app_dict.keys() if app_dict[w] >= MIN_APP_FREC])
