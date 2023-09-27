import numpy as np


def simple_hash_table(values_list: list, n_bins: int) -> dict:
    """
    function to compute hash value for an integer
    :param values_list: list of integers
    :param n_bins: number of bins in hashtable
    :return: hash table
    """

    def hash_function(value: int, n_bins: int) -> int:
        return int(value) % n_bins

    hash_table = {i: [] for i in range(n_bins)}
    for i in values_list:
        hash_value = hash_function(i, n_bins)
        hash_table[hash_value].append(i)

    return hash_table


def side_of_plane(P, v):
    """
    Return 1, 0 or -1 based on the side of the plane P the vector V lies
    :param P: reference Plane
    :param v: vector to get direction of
    :return: 1 if v is above P, -1 if v is below P, 0 if v lies on P
    """
    dot_product = np.dot(P, v.T)
    dot_product_sign = np.sign(dot_product)
    dot_product_scalar_sign = dot_product_sign.item()
    return dot_product_scalar_sign


def hash_function_multi_plane(p_list, v):
    """
    function to compute hash value of a vector w.r.t multiple planes
    :param p_list: List of planes
    :param v: Vector to get hash value of
    :return: hash-value of vector where it is localized w.r.t collection of planes
    """
    hash_value = 0
    for i, p in enumerate(p_list):
        sign = side_of_plane(p, v)
        hash_i = 1 if sign >= 0 else 0
        hash_value += (2 ** i) * hash_i
    return hash_value


def side_of_plane_matrix(plane_matrix, vector):
    """
    function to compute the direction of vector w.r.t planes in the matrix all together
    :param plane_matrix: matrix of planes
    :param vector: vector to get direction of
    :return: matrix of entries corresponds to direction of vector for every plane
    """
    dot_product = np.dot(plane_matrix, vector.T)
    dot_product_sign = np.sign(dot_product)
    return dot_product_sign


def hash_function_multi_plane_matrix(plane_matrix, vector):
    """
    compute hash value of a vector w.r.t matrix of planes
    :param plane_matrix:
    :param vector:
    :return:
    """
    sign_matrix = side_of_plane_matrix(plane_matrix, vector)
    hash_value = 0
    for i in range(plane_matrix.shape[0]):
        sign_i = sign_matrix[i].item()
        hash_i = 1 if sign_i >= 0 else 0
        hash_value += (2 ** i) * hash_i
    return hash_value
