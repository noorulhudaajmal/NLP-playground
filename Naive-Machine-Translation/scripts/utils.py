import numpy as np


def cosine_similarity(a, b):
    """
    compute cosine similarity between vector A and B
    :param a: vector A
    :param b: vector B
    :return: cosine similarity score
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)


def get_nearest_neighbour_vector(vector, candidate_vectors, k=1):
    """
    function to get the nearest similar vector in candidate vectors corpus for given vector
    :param vector: a word vector to get the nearest similar vector for
    :param candidate_vectors: the corpus of vectors to search for similar vectors
    :param k: specify the number of nearest neighbours
    :return: top nearest neighbour's indices from candidate corpus
    """
    similarity_list = [cosine_similarity(vector, i) for i in candidate_vectors]
    sorted_similarity_indices = np.argsort(similarity_list)
    return sorted_similarity_indices[-k:]
