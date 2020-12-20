import gzip
import pickle
from .paths import processed


with gzip.open(processed / 'reuters.pkl.gz', 'rb') as file:
    reuters = pickle.load(file)
