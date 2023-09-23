import json


def save(tokenizer, file_path):
    tokenizer_json = tokenizer.to_json()
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json_file.write(tokenizer_json)
