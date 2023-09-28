from nltk.corpus import twitter_samples
from pre_processing import *
from utils import *
import numpy as np

EMBEDDINGS_PATH = "data/en_embeddings.p"
N_DIMS = 300
N_PLANES = 10
N_UNIVERSES = 25
DOC_ID = 0

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
tweets = positive_tweets + negative_tweets

en_embeddings_subset = read_pickled_data(EMBEDDINGS_PATH)
ind2vec, doc_vec_matrix = get_document_vectors(tweets, en_embeddings_subset)

planes_list = [np.random.normal(size=(N_DIMS, N_PLANES)) for i in range(N_UNIVERSES)]

hash_tables = []
id_tables = []
for universe_id in range(N_UNIVERSES):  # there are 25 hashes
    planes = planes_list[universe_id]
    hash_table, id_table = make_hash_table(doc_vec_matrix, planes)
    hash_tables.append(hash_table)
    id_tables.append(id_table)

doc_to_search = tweets[DOC_ID]
vec_to_search = doc_vec_matrix[DOC_ID]

nearest_neighbor_ids = approximate_knn(DOC_ID, vec_to_search, planes_list, hash_tables, id_tables, k=3,
                                       num_universes_to_use=5)

print(f"Nearest neighbors for document {DOC_ID}")
print(f"Document contents: {doc_to_search}")
print("")

for neighbor_id in nearest_neighbor_ids:
    print(f"Nearest neighbor at document id {neighbor_id}")
    print(f"document contents: {tweets[neighbor_id]}")
