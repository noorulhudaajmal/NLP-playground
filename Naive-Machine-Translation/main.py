import pickle
from scripts.data_reader import read_pickled_data, get_embedding_matrix, read_dictionary
from model.translator import Translator

SEED = 101
SOURCE_EMBEDDINGS_PATH = "data/en_embeddings.p"
TARGET_EMBEDDINGS_PATH = "data/fr_embeddings.p"
MAPPING_DICT_PATH_TRAIN = 'data/en-fr.train.txt'
MAPPING_DICT_PATH_TEST = 'data/en-fr.test.txt'
LEARNING_RATE = 0.4
EPOCHS = 1000


en_embeddings_subset = read_pickled_data(SOURCE_EMBEDDINGS_PATH)
fr_embeddings_subset = read_pickled_data(TARGET_EMBEDDINGS_PATH)

en_fr_train = read_dictionary(MAPPING_DICT_PATH_TRAIN)
en_fr_test = read_dictionary(MAPPING_DICT_PATH_TEST)

X_train, Y_train = get_embedding_matrix(en_fr_train, en_embeddings_subset, fr_embeddings_subset)
X_val, Y_val = get_embedding_matrix(en_fr_test, en_embeddings_subset, fr_embeddings_subset)

model = Translator(X_train, Y_train, seed=SEED)
model.train_model(training_steps=EPOCHS, learning_rate=LEARNING_RATE, verbose=True)

transformation_matrix = model.get_transformation_matrix()
pickle.dump(transformation_matrix, open('model/R.pickle', 'wb'))

acc = model.test_model(X_val, Y_val)  # this might take a minute or two
print(f"accuracy on test set is {acc * 100:.2f}%")
