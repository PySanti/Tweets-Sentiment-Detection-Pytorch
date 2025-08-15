from utils.convert_base_X import convert_base_X
from utils.load_base_data import load_base_data
from utils.generate_corpus import generate_corpus



print("Cargando datos base")
base_dataset = load_base_data()
X = base_dataset.drop(labels=['Label'], axis=1)
Y = base_dataset['Label'].to_numpy()

print("Convirtiendo X base")
X = convert_base_X(X)

print("Generando corpus")
corpus = generate_corpus(X)

print(corpus)
print(len(corpus))
