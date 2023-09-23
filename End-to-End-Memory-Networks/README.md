# End-to-End Memory Networks

This repository contains a question-answering model based on the Memory Network architecture for solving tasks from the bAbI dataset. The model is implemented in Python using the Keras library and is designed to answer questions about short stories. The model used here follows the network architecture described in paper [End-To-End Memory Networks](https://arxiv.org/abs/1503.08895).

## Usage

### `main.py`

The `main.py` script serves as the entry point for the project. It performs the following tasks:

1. Loads and preprocesses the training and testing data from the specified paths.
2. Builds a vocabulary from the data.
3. Tokenizes the data using the `vectorize_data` function.
4. Builds and trains the question-answering model using the `MemoryNetwork` class.
5. Saves the trained model, evaluation plot, and test results.
6. Saves the tokenizer for future use.

You can customize the following constants in `main.py` to control the training and evaluation process:

- `TRAINING_DATA`: Path to the training data file.
- `TESTING_DATA`: Path to the testing data file.
- `EPOCHS`: Number of training epochs.
- `BATCH_SIZE`: Batch size for training.
- `TOKENIZER_PATH`: Path to save the tokenizer.
- `MODEL_PATH`: Path to save the trained model.
- `EVALUATION_PLOT_PATH`: Path to save the evaluation plot.
- `TEST_RESULTS_PATH`: Path to save the test results.

To run the project, execute the following command:

```bash
python main.py
```

### Jupyter Notebook

The Jupyter Notebook (`end-to-end-memory-networks.ipynb`) contains code and explanations for various parts of the project. It provides a step-by-step guide to data preprocessing, model building, training, evaluation, and testing.

## Requirements

Ensure you have the necessary Python packages installed by running:

```bash
pip install -r requirements.txt
```

## Data

The project uses the [bAbI](https://huggingface.co/datasets/facebook/babi_qa) dataset, specifically task QA2, which involves answering questions based on stories. The dataset is stored in the `data` directory.

## Models

The model architecture is defined in `models/model.py`, and the trained model is saved as `models/memory_model.keras`.

## Results

- Evaluation plots are saved as `results/evaluation_plot.png`.
- Test results are saved as `results/test_results.csv`.

## Scripts

The `scripts` directory contains the following scripts:

- `test.py`: Tests the trained model on a subset of the test data and displays the results.
- `evaluation.py`: Defines a class to plot the training and validation loss and accuracy over epochs.
- `data_preprocessing.py`: Contains functions for data parsing, vocabulary building, and data vectorization.

## Tokenizer

The tokenizer used for data vectorization is saved as `tokenizer/tokenizer.json`. You can load this tokenizer for future use.
