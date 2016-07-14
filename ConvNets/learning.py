import keras
import theano
import pickle

data = pickle.load(open("data_augmented", "rb"))
data = data['data']

