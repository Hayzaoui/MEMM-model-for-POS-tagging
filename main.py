import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test
import scipy

def main():
    threshold = 2
    lam = 0.3

    train_path = "data/train1.wtag"
    test_path = "data/test1.words"

    weights_path = 'weights1.pkl'
    predictions_path = 'predictions.wtag'

    statistics, feature2id = preprocess_train(train_path, threshold)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)


if __name__ == '__main__':
    main()