import streamlit as st
import numpy as np
from scripts.data_reader import read_pickled_data, get_embedding_matrix, read_dictionary
from scripts.utils import get_nearest_neighbour_vector

SOURCE_EMBEDDINGS_PATH = "data/en_embeddings.p"
TARGET_EMBEDDINGS_PATH = "data/fr_embeddings.p"
MAPPING_DICT_PATH_TRAIN = 'data/en-fr.train.txt'
MAPPING_DICT_PATH_TEST = 'data/en-fr.test.txt'
PICKLED_TRANSFORMATION_MATRIX_PATH = 'model/R.pickle'

en_embeddings_subset = read_pickled_data(SOURCE_EMBEDDINGS_PATH)
fr_embeddings_subset = read_pickled_data(TARGET_EMBEDDINGS_PATH)
en_fr_train = read_dictionary(MAPPING_DICT_PATH_TRAIN)
en_fr_test = read_dictionary(MAPPING_DICT_PATH_TEST)
en_fr_train.update(en_fr_test)
en_corpus = en_embeddings_subset.keys()

X, Y = get_embedding_matrix(en_fr_train, en_embeddings_subset, fr_embeddings_subset)
R = read_pickled_data(PICKLED_TRANSFORMATION_MATRIX_PATH)

st.title("NAIVE MACHINE TRANSLATION")
st.write("## ENGLISH to FRENCH translator")

input_word = st.text_input(label='English word')

if input_word in en_corpus:
    source_vector = en_embeddings_subset[input_word]
    predicted_vector = np.dot(source_vector, R)
    predicted_indices = get_nearest_neighbour_vector(predicted_vector, Y, k=3)
    candidate_vectors = [Y[index] for index in predicted_indices]
    target_words = []
    for key, value in fr_embeddings_subset.items():
        for i in candidate_vectors:
            if np.array_equal(value, i):
                target_words.append(key)
    st.info(f"### Translated words can be: \n{'  |  '.join(target_words)}")

elif input_word is not None and input_word != '':
    st.warning('Sorry, word not in English corpus!!!')
