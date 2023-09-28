import re
import string
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def read_pickled_data(filepath: str):
    """
    load and reads the pickled data from specified path
    :param filepath: filepath to rad data from
    :return: data retrieved from pickled file
    """
    return pickle.load(open(filepath, 'rb'))


def process_text(text: str):
    """
    tokenize and clean the input string
    :param text: the raw text
    :return: clean tokenized text
    """
    stemmer = PorterStemmer()
    stop_words = stopwords.words('english')

    text = re.sub(r'^RT[\s]', '', text)
    text = re.sub(r'\$\w*', '', text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'#', '', text)
    tokens = word_tokenize(text)
    text_clean = [stemmer.stem(word) for word in tokens if (word not in stop_words and word not in string.punctuation)]
    return text_clean


def get_document_embeddings(text: str, embeddings_dict: dict):
    """
    function to retrieve and compute the document embedding vector
    :param text: raw text document
    :param embeddings_dict: dictionary with words as keys and values as their corresponding embeddings
    :return: embedding vector of the document
    """
    doc_embedding = np.zeros(300)
    processed_text = process_text(text)
    for token in processed_text:
        doc_embedding += embeddings_dict.get(token, 0)
    return doc_embedding


def get_document_vectors(documents, embeddings_dict):
    """
    function to convert list of text documents into their vector embedding matrix
    :param documents: list of raw text documents
    :param embeddings_dict: dictionary with words as keys and values as their corresponding embeddings
    :return: dictionary with index to document vector embedding, document vectors embedding matrix
    """
    index_to_doc = dict()
    document_vectors = []

    for i, doc in enumerate(documents):
        doc_embeddings = get_document_embeddings(doc, embeddings_dict)
        index_to_doc[i] = doc_embeddings
        document_vectors.append(doc_embeddings)
    document_vectors_matrix = np.vstack(document_vectors)

    return index_to_doc, document_vectors_matrix
