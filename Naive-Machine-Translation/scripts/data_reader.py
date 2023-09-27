import pickle
from typing import Dict, Any
import numpy as np
import pandas as pd


def read_pickled_data(filepath: str) -> Any:
    """
    load and reads the pickled data from specified path
    :param filepath: filepath to rad data from
    :return: data retrieved from pickled file
    """
    return pickle.load(open(filepath, 'rb'))


def read_dictionary(filepath: str) -> dict[str, str]:
    file = pd.read_csv(filepath, delimiter=' ')
    return dict((source, target) for index, source, target in file.itertuples())


def get_embedding_matrix(mapping_dic: dict,
                         source_embeddings: dict,
                         target_embeddings: dict
                         ) -> (np.ndarray, np.ndarray):
    """
    function to produce matrix of embedding of both source and target vectors
    :param mapping_dic: dictionary with keys of source language and corresponding target language values
    :param source_embeddings: dictionary mapping of source language words to their embeddings
    :param target_embeddings: dictionary mapping of target language words to their embeddings
    :return: X->matrix with embedding of source language, Y->matrix with embeddings of target language
    """

    source_embedding_vectors = list()
    target_embedding_vectors = list()

    source_corpus = source_embeddings.keys()
    target_corpus = target_embeddings.keys()

    for source, target in mapping_dic.items():
        if source in source_corpus and target in target_corpus:
            source_embedding_vectors.append(source_embeddings[source])
            target_embedding_vectors.append(target_embeddings[target])

    X = np.vstack(source_embedding_vectors)
    Y = np.vstack(target_embedding_vectors)

    return X, Y

