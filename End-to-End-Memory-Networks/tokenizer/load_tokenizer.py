from keras.preprocessing.text import tokenizer_from_json


def load(file_path):
    # Load the tokenizer from the JSON file
    with open(file_path, 'r', encoding='utf-8') as json_file:
        tokenizer_json = json_file.read()
        tokenizer = tokenizer_from_json(tokenizer_json)

    return tokenizer
