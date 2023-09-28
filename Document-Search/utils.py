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


def hash_value_of_vector(v, planes):
    """
    Create a hash for a vector; hash_id says which random hash to use.
    :param v: document vector
    :param planes: the set of planes that divide up the region
    :return: a number which is used as a hash for vector
    """
    dot_product = np.dot(v, planes)
    sign_of_dot_product = np.sign(dot_product)
    h = sign_of_dot_product >= 0
    h = np.squeeze(h)

    hash_value = 0
    n_planes = planes.shape[1]
    for i in range(n_planes):
        # 2^i * h_i
        hash_value += np.power(2, i) * h[i]

    hash_value = int(hash_value)
    return hash_value


def make_hash_table(vecs, planes):
    """
    function to create a hash table
    :param vecs: list of vectors to be hashed
    :param planes: the matrix of planes in a single "universe", with shape (embedding dimensions, number of planes)
    :return: (dictionary - keys are hashes, values are lists of vectors (hash buckets),  dictionary - keys are hashes, values are list of vectors id's_
    """
    num_of_planes = planes.shape[1]
    num_buckets = 2**num_of_planes
    hash_table = {i:[] for i in range(num_buckets)}
    id_table = {i:[] for i in range(num_buckets)}

    for i, v in enumerate(vecs):
        h = hash_value_of_vector(v,planes)
        hash_table[h].append(v)
        id_table[h].append(i)

    return hash_table, id_table


def approximate_knn(doc_id, v, planes_l, hash_tables, id_tables, k=1, num_universes_to_use=25):
    """Search for k-NN using hashes."""
    assert num_universes_to_use <= 25
    vecs_to_consider_l = list()
    ids_to_consider_l = list()
    ids_to_consider_set = set()

    for universe_id in range(num_universes_to_use):
        planes = planes_l[universe_id]
        hash_value = hash_value_of_vector(v, planes)
        hash_table = hash_tables[universe_id]
        document_vectors_l = hash_table[hash_value]
        id_table = id_tables[universe_id]
        new_ids_to_consider = id_table[hash_value]
        if doc_id in new_ids_to_consider:
            new_ids_to_consider.remove(doc_id)
            print(f"removed doc_id {doc_id} of input vector from new_ids_to_search")
        for i, new_id in enumerate(new_ids_to_consider):
            if new_id not in ids_to_consider_set:
                document_vector_at_i = document_vectors_l[i]
                vecs_to_consider_l.append(document_vector_at_i)
                ids_to_consider_l.append(new_id)
                ids_to_consider_set.add(new_id)

    print("Fast considering %d vecs" % len(vecs_to_consider_l))
    vecs_to_consider_arr = np.array(vecs_to_consider_l)
    nearest_neighbor_idx_l = get_nearest_neighbour_vector(v, vecs_to_consider_arr, k=k)
    nearest_neighbor_ids = [ids_to_consider_l[idx]
                            for idx in nearest_neighbor_idx_l]

    return nearest_neighbor_ids
