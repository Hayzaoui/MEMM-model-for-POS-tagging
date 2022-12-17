from preprocessing import read_test, represent_input_with_features
from tqdm import tqdm
from collections import OrderedDict
import itertools
import math, numpy as np


def memm_viterbi(sentence, pre_trained_weights, feature2id, beam):
    """
      Write your MEMM Viterbi implementation below
      You can implement Beam Search to improve runtime
      Implement q efficiently (refer to conditional probability definition in MEMM slides)
      """
    all_tags = set(list(feature2id.feature_statistics.tags) + ["*"])
    pi = OrderedDict()
    bpi = OrderedDict()
    pi[(0, '*', '*')] = 1
    predicted_tags = ['*'] * (len(sentence) - 2)
    pptag_dict = OrderedDict()
    ptag_dict = OrderedDict()
    pptag_dict['*'] = 1
    ptag_dict['*'] = 1
    for k, cur_word in enumerate(sentence):
        pptag_list = sorted(pptag_dict, key=pptag_dict.get, reverse=True)[:beam]
        ptag_list = sorted(ptag_dict, key=ptag_dict.get, reverse=True)[:beam]
        pptag_dict = OrderedDict()
        ptag_dict = OrderedDict()
        if k == 0:
          pptag_dict['*'] = 1
          ptag_dict['*'] = 1
          continue
        if cur_word == '~':
          continue
        prev_word, pp_word, next_word = sentence[k - 1], sentence[k - 2], sentence[k + 1]
        if k == 1:
          pp_tag, prev_tag = '*', '*'
          for cur_tag in all_tags:
            cur_history = (cur_word, cur_tag, prev_word, prev_tag, pp_word, pp_tag, next_word)
            possible_histories = []
            for tag in all_tags:
              possible_histories.append((cur_word, tag, prev_word, prev_tag, pp_word, pp_tag, next_word))
            features_idx = represent_input_with_features(history=cur_history,
                                                         dict_of_dicts=feature2id.feature_to_idx)
            count_features_weight = 0
            for idx in features_idx:
              count_features_weight += pre_trained_weights[idx]
            q_1 = math.exp(count_features_weight)
            q_2 = 0
            for history in possible_histories:
              count_features_weight = 0
              features_idx = represent_input_with_features(history=history,
                                                           dict_of_dicts=feature2id.feature_to_idx)
              for idx in features_idx:
                count_features_weight += pre_trained_weights[idx]
              q_2 += math.exp(count_features_weight)

            pi[(k, prev_tag, cur_tag)] = q_1 / q_2

        for cur_tag in all_tags:
          curent_tag = cur_tag
          plist = ptag_list if k > 3 else all_tags
          for prev_tag in plist:
            count_features_weight = 0
            probabilities = {tag: 0 for tag in all_tags}

            for j, pp_tag in enumerate(pptag_list):
              if (k - 1, pp_tag, prev_tag) not in pi.keys() or cur_tag == '*':
                # pi[(k - 1, pp_tag, prev_tag)] = 0

                continue
              cur_history = (cur_word, cur_tag, prev_word, prev_tag, pp_word, pp_tag, next_word)
              possible_histories = []
              for tag in all_tags:
                possible_histories.append((cur_word, tag, prev_word, prev_tag, pp_word, pp_tag, next_word))
              features_idx = represent_input_with_features(history=cur_history,
                                                           dict_of_dicts=feature2id.feature_to_idx)
              for idx in features_idx:
                count_features_weight += pre_trained_weights[idx]
              q_1 = math.exp(count_features_weight)
              q_2 = 0
              for history in possible_histories:
                count_features_weight = 0
                features_idx = represent_input_with_features(history=history,
                                                             dict_of_dicts=feature2id.feature_to_idx)
                for idx in features_idx:
                  count_features_weight += pre_trained_weights[idx]
                q_2 += math.exp(count_features_weight)

              Q = q_1 / q_2

              probabilities[pp_tag] = pi[(k - 1, pp_tag, prev_tag)] * Q

            key_max = ''
            max_pikuv = 0

            for key, val in probabilities.items():
              if val > max_pikuv:
                max_pikuv = val
                key_max = key

            pi[(k, prev_tag, cur_tag)] = max_pikuv
            bpi[(k, prev_tag, cur_tag)] = key_max
            pptag_dict[prev_tag] = pi[(k, prev_tag, cur_tag)]
            ptag_dict[cur_tag] = pi[(k, prev_tag, cur_tag)]
    n = len(sentence) - 2
    maxx = 0
    u, v = '', ''
    for key, val in pi.items():
        if key[0] == n - 1:
          if val > maxx:
            u, v = key[1], key[2]
            maxx = val
    predicted_tags[n - 1] = v
    predicted_tags[n - 2] = u
    for k in range(n - 3, 0, -1):
      tkp1 = predicted_tags[k + 1]
      tkp2 = predicted_tags[k + 2]

      predicted_tags[k] = bpi[(k + 2, tkp1, tkp2)]

    return predicted_tags


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)
    output_file = open(predictions_path, "a+")

    if tagged:
        all_tags = list(feature2id.feature_to_idx['f105'])
        lengh_all_tags = len(all_tags)
        dict_tags = {i: all_tags[i][0] for i in range(lengh_all_tags)}
        dict_tags_inv = {v: k for k, v in dict_tags.items()}
        confusion_matrix = np.zeros((lengh_all_tags, lengh_all_tags))

        m = 0
        accuracy_mean = 0

        for k, sen in tqdm(enumerate(test), total=len(test)):
            sentence = sen[0]
            correct_tags = sen[1]
            correct_tags = correct_tags[2:-2]
            pred = memm_viterbi(sentence, pre_trained_weights, feature2id, 6)[2:]
            sentence = sentence[2:]
            accuracy = 0
            n = 0

            for i in range(len(correct_tags)):
                if correct_tags[i] == pred[i]:
                  accuracy += 1
                  accuracy_mean += 1
                n += 1
                m += 1
            print("Accuracy:", round(float(accuracy / n), 6))
            print("Accuracy Mean :", round(float(accuracy_mean / m), 6))

            for i in range(len(pred)):
                predicted = dict_tags_inv[pred[i]]
                try:
                    true = dict_tags_inv[correct_tags[i]]
                    confusion_matrix[predicted, true] += 1
                except KeyError:
                    pass

            for i in range(len(pred)):
                if i > 0:
                    output_file.write(" ")
                output_file.write(f"{sentence[i]}_{pred[i]}")
            output_file.write("\n")
        output_file.close()

        for_worst = confusion_matrix - np.diag(np.diag(confusion_matrix))
        # Selecting ten worst tags
        ten_worst = np.argsort(np.sum(for_worst, axis=0))[-10:]
        ten_worst_conf_mat = confusion_matrix[np.ix_(ten_worst, ten_worst)]
        # Computing Accuracy
        accuracy = 100 * np.trace(confusion_matrix) / np.sum(confusion_matrix)
        # print("Model " + str(num_model) + " Accuracy.: " + str(accuracy) + " %")
        print("Accuracy.: " + str(accuracy) + " %")
        print("Ten Worst Elements: " + str([dict_tags[i] for i in ten_worst]))
        print("Confusion Matrix:")
        print(ten_worst_conf_mat)

    else:
        for k, sen in tqdm(enumerate(test), total=len(test)):
            sentence = sen[0]
            pred = memm_viterbi(sentence, pre_trained_weights, feature2id, beam=3)[2:]
            sentence = sentence[2:]

            for i in range(len(pred)):
                if i > 0:
                    output_file.write(" ")
                output_file.write(f"{sentence[i]}_{pred[i]}")
            output_file.write('._.')
            output_file.write("\n")
    output_file.close()
