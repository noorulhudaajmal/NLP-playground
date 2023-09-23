import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pandas as pd


class ModelTester:
    def __init__(self, model_weights_path, vocab, tokenizer, max_story_len, max_question_len):
        self.model_weights_path = model_weights_path
        self.vocab = vocab
        self.max_story_len = max_story_len
        self.max_question_len = max_question_len
        self.tokenizer = tokenizer

        # Load the trained model
        self.model = load_model(self.model_weights_path)

    def vectorize_story_query(self, story, query):
        story_seq = pad_sequences([[self.tokenizer.word_index[word.lower()] for word in story]], maxlen=self.max_story_len)
        query_seq = pad_sequences([[self.tokenizer.word_index[word.lower()] for word in query]], maxlen=self.max_question_len)
        return story_seq, query_seq

    def test_model(self, test_data):
        results = []
        for i in test_data:
            story, query, answer = i
            story_seq, query_seq = self.vectorize_story_query(story, query)
            prediction = self.model.predict([story_seq, query_seq], verbose=0)
            predicted_index = np.argmax(prediction)
            predicted_answer = self.tokenizer.index_word[predicted_index]
            results.append([' '.join(story), ' '.join(query), answer, predicted_answer])
        df = pd.DataFrame(results, columns=['Story', 'Query', 'Answer', 'Predicted Answer'])
        return df
