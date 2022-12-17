import pickle
from inference import tag_all_test


def generate_comp_tag(weights_path, comp_mi_path, predictions_path):
    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]
    tag_all_test(comp_mi_path, pre_trained_weights, feature2id, predictions_path)

if __name__ == '__main__':
    weights_comp1_path = "weight1.pkl"
    weights_comp2_path = "weight_comp2.pkl"

    comp_m1_path = 'comp1.words'
    comp_m2_path = 'comp2.words'
    predictions_path_1 = 'comp_m1_345123624_931189518.wtag'
    predictions_path_2 = 'comp_m2_345123624_931189518.wtag'
    generate_comp_tag(weights_comp1_path, comp_m1_path, predictions_path_1)  # model train with train1
    generate_comp_tag(weights_comp2_path, comp_m2_path, predictions_path_2)  # model train with train2
