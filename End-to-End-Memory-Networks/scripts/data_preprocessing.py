import numpy as np
import nltk
# nltk.download("punkt")
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def flatten(text_list):
    return [j for i in text_list for j in i]


def data_parser(data_path):
    with open(data_path, 'rb') as f:
        data = f.read()

    data = data.decode('utf-8').split("\n")
    extracted_data = []
    last_visited_story = []

    for line in data:
        if line != "":
            story_line_num, story_line = line.split(' ', 1)
            story_line_num = int(story_line_num)

            if story_line_num == 1:
                last_visited_story = []

            if '\t' in story_line:
                query, answer, supported_fact_id = story_line.split("\t")
                story_lines = [i for i in last_visited_story if i != None]
                story_lines = flatten(story_lines)
                extracted_data.append((story_lines, word_tokenize(query), answer))
            else:
                last_visited_story.append(word_tokenize(story_line))

    return extracted_data


def build_vocab(train_data, test_data):
    all_data = test_data + train_data
    vocab = set()
    for story, question, answer in all_data:
        vocab = vocab.union(set(story))
        vocab = vocab.union(set(question))
    return vocab


def vectorize_stories(data, word_index, max_story_len, max_question_len):
    X = []
    Xq = []
    Y = []

    for story, question, answer in data:
        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in question]
        y = np.zeros(len(word_index) + 1)
        y[word_index[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)

    return (pad_sequences(X, maxlen=max_story_len), pad_sequences(Xq, maxlen=max_question_len), np.array(Y))


def vectorize_data(vocab):
    tokenizer = Tokenizer(filters=[])
    tokenizer.fit_on_texts(vocab)
    return tokenizer