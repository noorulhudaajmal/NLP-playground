from scripts.data_preprocessing import data_parser, build_vocab, vectorize_stories, vectorize_data
from models.model import MemoryNetwork
from scripts.evaluation import ModelHistoryPlotter
from scripts.test import ModelTester
from tokenizer import load_tokenizer, save_tokenizer


TRAINING_DATA = "data/tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_train.txt"
TESTING_DATA = "data/tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_test.txt"

EPOCHS = 150
BATCH_SIZE = 32

TOKENIZER_PATH = 'tokenizer/tokenizer.json'
MODEL_PATH = 'models/memory_model.keras'
EVALUATION_PLOT_PATH = 'results/evaluation_plot.png'
TEST_RESULTS_PATH = 'results/test_results.csv'


def main():
    # Paths to data files
    train_data_path = TRAINING_DATA
    test_data_path = TESTING_DATA

    # Parsing the data
    train_data = data_parser(train_data_path)
    test_data = data_parser(test_data_path)

    # Building vocabulary
    vocab = build_vocab(train_data, test_data)
    vocab_len = len(vocab) + 1

    # getting individual lengths of stories and queries
    all_stories_len = [len(data[0]) for data in train_data + test_data]
    all_questions_len = [len(data[1]) for data in train_data + test_data]

    # LENGTH OF LONGEST STORY & QUERY
    max_story_len = max(all_stories_len)
    max_question_len = max(all_questions_len)

    # Tokenizing
    tokenizer = vectorize_data(vocab)

    # Vectorizing data
    inputs_train, queries_train, answers_train = vectorize_stories(train_data, tokenizer.word_index, max_story_len, max_question_len)
    inputs_test, queries_test, answers_test = vectorize_stories(test_data, tokenizer.word_index, max_story_len, max_question_len)

    # Building and training the question-answering model
    mem_model = MemoryNetwork(max_story_len, max_question_len, vocab_len)
    print("\n\nMODEL TRAINING......")
    mem_model.train(X=[inputs_train, queries_train],
                    y=answers_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=([inputs_test, queries_test], answers_test))
    model = mem_model.get_model()
    mem_model.save_model()

    print("\nMODEL HISTORY...")
    plotter = ModelHistoryPlotter(model)
    acc_loss_plot = plotter.plot_history()

    acc_loss_plot.savefig(EVALUATION_PLOT_PATH)

    print("\nTESTING....")
    tester = ModelTester(
        model_weights_path=MODEL_PATH,
        vocab=vocab,
        tokenizer=tokenizer,
        max_story_len=max_story_len,
        max_question_len=max_question_len
    )

    # Test the model
    test_results = tester.test_model(test_data)
    test_results.to_csv(TEST_RESULTS_PATH)
    print(f"Test results saved in df: {TEST_RESULTS_PATH}")

    # Save the tokenizer
    save_tokenizer.save(tokenizer, TOKENIZER_PATH)

if __name__ == "__main__":
    main()
