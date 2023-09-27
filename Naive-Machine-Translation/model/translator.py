import numpy as np
from scripts.utils import get_nearest_neighbour_vector


class Translator:
    def __init__(self, source_embedding_matrix, target_embedding_matrix, seed=101) -> None:
        """
        initiate the model params
        :param source_embedding_matrix: a matrix of dimension (m,n) where the columns are the source language embeddings
        :param target_embedding_matrix: a matrix of dimension (m,n) where the columns correspond to the target language embeddings
        """
        self.X = source_embedding_matrix
        self.Y = target_embedding_matrix

        np.random.seed(seed)
        # a matrix of dimension (n,n) - transformation matrix from source to target language vector space embeddings
        self.R = np.random.rand(source_embedding_matrix.shape[1], source_embedding_matrix.shape[1])

    def compute_loss(self):
        """
        function to compute loss as defined by minimizing Frobenius norm of XR-Y
        :return: computed loss for current X,Y, and R
        """
        m = self.X.shape[0]
        norm = np.linalg.norm(np.dot(self.X, self.R) - self.Y)
        squared_norm = norm ** 2
        return squared_norm / m

    def compute_gradient(self):
        """
        function to compute gradient of the loss function
        :return: gradient based on current X,Y and R
        """
        m = self.X.shape[0]
        gradient = (2 / m) * np.dot(np.transpose(self.X), np.dot(self.X, self.R) - self.Y)
        return gradient

    def train_model(self, training_steps: int, learning_rate: float, verbose: bool):
        """
        function to optimize the transformation matrix using gradient descent
        :param training_steps: describes how many steps will gradient descent algorithm do
        :param learning_rate: describes how big steps will  gradient descent algorithm do
        :param verbose: specify whether the loss info to be printed or not
        :return: None
        """
        for i in range(training_steps):
            if verbose and i % 10 == 0:
                print(f'Iteration: {i}\t Loss: {self.compute_loss():.4f}')
            gradient = self.compute_gradient()
            self.R -= learning_rate * gradient

    def predict(self, source_vector):
        """
        predict target vector
        :param source_vector: the vector to be transformed
        :return: the index of the nearest similar vector in target corpus
        """
        predicted_target_vector = np.dot(source_vector, self.R)
        predicted_index = get_nearest_neighbour_vector(predicted_target_vector, self.Y)
        return self.Y[predicted_index[0]]

    def test_model(self, source_embeddings, target_embeddings):
        """
        test how well does transformation matrix, R is optimized
        :param source_embeddings: source language embeddings matrix for test
        :param target_embeddings: target language embeddings matrix for test
        :return: accuracy of the predictions by model on test data
        """
        predicted_target_vectors = np.dot(source_embeddings, self.R)
        correct_prediction_count = 0

        for i in range(len(predicted_target_vectors)):
            predicted_indices = get_nearest_neighbour_vector(predicted_target_vectors[i], target_embeddings)
            if predicted_indices == i:
                correct_prediction_count += 1

        return correct_prediction_count / len(predicted_target_vectors)

    def get_transformation_matrix(self):
        """
        getter for R
        :return: R, the transformation matrix
        """
        return self.R
