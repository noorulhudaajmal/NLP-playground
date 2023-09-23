from keras.layers import Input, Embedding, LSTM, Dense, concatenate, Dropout, Activation, Permute, add, dot
from keras.models import Model, Sequential


class MemoryNetwork:
    def __init__(self, max_story_len, max_question_len, vocab_len):
        self.max_story_len = max_story_len
        self.max_question_len = max_question_len
        self.vocab_len = vocab_len
        self.model = self.build_memory_network()

    def build_memory_network(self):
        input_sequence = Input((self.max_story_len,))
        question = Input((self.max_question_len,))

        input_encoder_m = Sequential()
        input_encoder_c = Sequential()
        question_encoder = Sequential()

        input_encoder_m.add(Embedding(input_dim=self.vocab_len, output_dim=64))
        input_encoder_m.add(Dropout(0.3))

        input_encoder_c.add(Embedding(input_dim=self.vocab_len, output_dim=self.max_question_len))
        input_encoder_c.add(Dropout(0.3))

        question_encoder.add(Embedding(input_dim=self.vocab_len, output_dim=64, input_length=self.max_question_len))
        question_encoder.add(Dropout(0.3))

        input_encoded_m = input_encoder_m(input_sequence)
        input_encoded_c = input_encoder_c(input_sequence)
        question_encoded = question_encoder(question)

        match = dot([input_encoded_m, question_encoded], axes=(2, 2))
        match = Activation('softmax')(match)

        response = add([match, input_encoded_c])
        response = Permute((2, 1))(response)

        answer = concatenate([response, question_encoded])

        answer = LSTM(32)(answer)
        answer = Dropout(0.3)(answer)
        answer = Dense(self.vocab_len)(answer)

        answer = Activation('softmax')(answer)

        model = Model(inputs=[input_sequence, question], outputs=answer)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def get_model(self):
        return self.model

    def train(self, X, y, batch_size, epochs, validation_data):
        self.model.fit(X, y, batch_size, epochs, validation_data)

    def save_model(self, file_path='models/memory_model.keras'):
        self.model.save(file_path)




