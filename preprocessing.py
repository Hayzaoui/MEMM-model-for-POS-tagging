from scipy import sparse
from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple
import pprint

WORD = 0
TAG = 1


class FeatureStatistics:
  def __init__(self):
    self.n_total_features = 0  # Total number of features accumulated

    # Init all features dictionaries
    feature_dict_list = ["f100", "f101", "f102", "f103", "f104", "f105", "f106", "f107", "f108", "f109", "f110",
                         "f111", "f112", "f113", "f114", "f115", "f116", "f117", "f118", "f119",
                         "f120"]  # the feature classes used in the code
    self.feature_rep_dict = {fd: OrderedDict() for fd in feature_dict_list}
    '''
        A dictionary containing the counts of each data regarding a feature class. For example in f100, would contain
        the number of times each (word, tag) pair appeared in the text.
        '''
    self.tags = set()  # a set of all the seen tags
    self.tags.add("~")
    self.tags_counts = defaultdict(int)  # a dictionary with the number of times each tag appeared in the text
    self.words_count = defaultdict(int)  # a dictionary with the number of times each word appeared in the text
    self.histories = []  # a list of all the histories seen at the test

  def get_word_tag_pair_count(self, file_path) -> None:
    """
            Extract out of text all word/tag pairs
            @param: file_path: full path of the file to read
            Updates the histories list
        """

    with open(file_path) as file:
        for line in file:
            if line[-1:] == "\n":
              line = line[:-1]
            split_words = line.split(' ')

            for word_idx in range(len(split_words)):

              cur_word, cur_tag = split_words[word_idx].split('_')

              if word_idx == len(split_words) - 1:
                next_word = '~'
              else:
                next_word, _ = split_words[word_idx + 1].split('_')
              if word_idx == 0:
                prev_word, prev_tag = ['*', '*']
                pp_word, pp_tag = ['*', '*']
              else:
                prev_word, prev_tag = split_words[word_idx - 1].split('_')
              if word_idx == 1:
                pp_word, pp_tag = ['*', '*']
              else:
                if word_idx != 0:
                  pp_word, pp_tag = split_words[word_idx - 2].split('_')
              self.tags.add(cur_tag)
              self.tags_counts[cur_tag] += 1
              self.words_count[cur_word] += 1
              # f100
              if (cur_word, cur_tag) not in self.feature_rep_dict["f100"]:
                self.feature_rep_dict["f100"][(cur_word, cur_tag)] = 1
              else:
                self.feature_rep_dict["f100"][(cur_word, cur_tag)] += 1

              len_word = len(cur_word)
              for i in range(1, 5):
                if len_word > i + 1:  # because there is no purpose to suffix with words of len 3
                  suf = cur_word[len_word - i:]
                  pref = cur_word[:i]
                  # f101
                  if (suf, cur_tag) not in self.feature_rep_dict["f101"]:
                    self.feature_rep_dict["f101"][(suf, cur_tag)] = 1
                  else:
                    self.feature_rep_dict["f101"][(suf, cur_tag)] += 1
                  # f102
                  if (pref, cur_tag) not in self.feature_rep_dict["f102"]:
                    self.feature_rep_dict["f102"][(pref, cur_tag)] = 1
                  else:
                    self.feature_rep_dict["f102"][(pref, cur_tag)] += 1
              # f103
              if (pp_tag, prev_tag, cur_tag) not in self.feature_rep_dict["f103"]:
                self.feature_rep_dict["f103"][(pp_tag, prev_tag, cur_tag)] = 1
              else:
                self.feature_rep_dict["f103"][(pp_tag, prev_tag, cur_tag)] += 1
              # f104
              if (prev_tag, cur_tag) not in self.feature_rep_dict["f104"]:
                self.feature_rep_dict["f104"][(prev_tag, cur_tag)] = 1
              else:
                self.feature_rep_dict["f104"][(prev_tag, cur_tag)] += 1
              # f105
              if (cur_tag,) not in self.feature_rep_dict["f105"]:
                self.feature_rep_dict["f105"][(cur_tag,)] = 1
              else:
                self.feature_rep_dict["f105"][(cur_tag,)] += 1
              # f106
              if (prev_word, cur_tag) not in self.feature_rep_dict["f106"]:
                self.feature_rep_dict["f106"][(prev_word, cur_tag)] = 1
              else:
                self.feature_rep_dict["f106"][(prev_word, cur_tag)] += 1
              # f107
              if (next_word, cur_tag) not in self.feature_rep_dict["f107"]:
                self.feature_rep_dict["f107"][(next_word, cur_tag)] = 1
              else:
                self.feature_rep_dict["f107"][(next_word, cur_tag)] += 1
              # f108
              if any(char.isdigit() for char in cur_word):
                if (cur_tag,) not in self.feature_rep_dict["f108"]:
                  self.feature_rep_dict["f108"][(cur_tag,)] = 1
                else:
                  self.feature_rep_dict["f108"][(cur_tag,)] += 1
              # f109
              if any(char.isupper() for char in cur_word):
                if (cur_tag,) not in self.feature_rep_dict["f109"]:
                  self.feature_rep_dict["f109"][(cur_tag,)] = 1
                else:
                  self.feature_rep_dict["f109"][(cur_tag,)] += 1
              # f110
              if all(char.isupper() for char in cur_word):
                if (cur_tag,) not in self.feature_rep_dict["f110"]:
                  self.feature_rep_dict["f110"][(cur_tag,)] = 1
                else:
                  self.feature_rep_dict["f110"][(cur_tag,)] += 1
              # f111
              number_like = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "tens",
                             "eleven", "twelve", "twenty", "fifty", "hundred", "hundreds", "thousand", "thousands"]
              if cur_word in number_like:
                if (cur_tag,) not in self.feature_rep_dict["f111"]:
                  self.feature_rep_dict["f111"][(cur_tag,)] = 1
                else:
                  self.feature_rep_dict["f111"][(cur_tag,)] += 1
              # f112
              if any(char.isupper() for char in cur_word) and any(char.isdigit() for char in cur_word) and any(
                char == '-' for char in cur_word):
                if (cur_tag,) not in self.feature_rep_dict["f112"]:
                  self.feature_rep_dict["f112"][(cur_tag,)] = 1
                else:
                  self.feature_rep_dict["f112"][(cur_tag,)] += 1
              # f113
              if '-' in cur_word:
                if (cur_tag,) not in self.feature_rep_dict["f113"]:
                  self.feature_rep_dict["f113"][(cur_tag,)] = 1
                else:
                  self.feature_rep_dict["f113"][(cur_tag,)] += 1
              # f114
              if cur_word[-4:] == 'ness' and cur_tag == 'NN':
                if (cur_word, cur_tag,) not in self.feature_rep_dict["f114"]:
                  self.feature_rep_dict["f114"][(cur_word, cur_tag,)] = 1
                else:
                  self.feature_rep_dict["f114"][(cur_word, cur_tag,)] += 1
              """# f115
              if prev_tag == 'NN' and cur_tag == 'JJ':
                if (prev_tag, cur_tag) not in self.feature_rep_dict["f115"]:
                  self.feature_rep_dict["f115"][(prev_tag, cur_tag)] = 1
                else:
                  self.feature_rep_dict["f115"][(prev_tag, cur_tag)] += 1
              # f116
              if prev_tag == 'JJ' and cur_tag == 'NN':
                if (prev_tag, cur_tag) not in self.feature_rep_dict["f116"]:
                  self.feature_rep_dict["f116"][(prev_tag, cur_tag)] = 1
                else:
                  self.feature_rep_dict["f116"][(prev_tag, cur_tag)] += 1"""
              """# f117
              if cur_word[-5:] == 'kappa' and cur_tag == 'NN':
                if (cur_word, cur_tag,) not in self.feature_rep_dict["f117"]:
                  self.feature_rep_dict["f117"][(cur_word, cur_tag,)] = 1
                else:
                  self.feature_rep_dict["f117"][(cur_word, cur_tag,)] += 1
              # f118
              if cur_word[-5:] == 'virus' and cur_tag == 'NN':
                if (cur_word, cur_tag,) not in self.feature_rep_dict["f118"]:
                  self.feature_rep_dict["f118"][(cur_word, cur_tag,)] = 1
                else:
                  self.feature_rep_dict["f118"][(cur_word, cur_tag,)] += 1
              # f119
              if cur_word[-2:] == 'ol' and cur_tag == 'NN':
                if (cur_word, cur_tag,) not in self.feature_rep_dict["f119"]:
                  self.feature_rep_dict["f119"][(cur_word, cur_tag,)] = 1
                else:
                  self.feature_rep_dict["f119"][(cur_word, cur_tag,)] += 1
              # f120
              if cur_word[-2:] == 'an' and cur_tag == 'NN':
                if (cur_word, cur_tag,) not in self.feature_rep_dict["f120"]:
                  self.feature_rep_dict["f120"][(cur_word, cur_tag,)] = 1
                else:
                  self.feature_rep_dict["f120"][(cur_word, cur_tag,)] += 1"""

            sentence = [("*", "*"), ("*", "*")]
            for pair in split_words:
                sentence.append(tuple(pair.split("_")))
            sentence.append(("~", "~"))

            for i in range(2, len(sentence) - 1):
              history = (
                sentence[i][0], sentence[i][1], sentence[i - 1][0], sentence[i - 1][1], sentence[i - 2][0],
                sentence[i - 2][1], sentence[i + 1][0])

              self.histories.append(history)


class Feature2id:
  def __init__(self, feature_statistics: FeatureStatistics, threshold: int):
    """
        @param feature_statistics: the feature statistics object
        @param threshold: the minimal number of appearances a feature should have to be taken
        """
    self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
    self.threshold = threshold  # feature count threshold - empirical count must be higher than this

    self.n_total_features = 0  # Total number of features accumulated

    # Init all features dictionaries
    self.feature_to_idx = {
      "f100": OrderedDict(), "f101": OrderedDict(), "f102": OrderedDict(), "f103": OrderedDict(),
      "f104": OrderedDict(), "f105": OrderedDict(), "f106": OrderedDict(), "f107": OrderedDict(),
      "f108": OrderedDict(), "f109": OrderedDict(), "f110": OrderedDict(), "f111": OrderedDict(),
      "f112": OrderedDict(), "f113": OrderedDict(), "f114": OrderedDict(), "f115": OrderedDict(),
      "f116": OrderedDict(), "f117": OrderedDict(), "f118": OrderedDict(), "f119": OrderedDict(),
      "f120": OrderedDict(),
    }
    self.represent_input_with_features = OrderedDict()

    self.histories_matrix = OrderedDict()
    self.histories_features = OrderedDict()
    self.small_matrix = sparse.csr_matrix
    self.big_matrix = sparse.csr_matrix

  def get_features_idx(self) -> None:
    """
        Assigns each feature that appeared enough time in the train files an idx.
        Saves those indices to self.feature_to_idx
        """
    for feat_class in self.feature_statistics.feature_rep_dict:
      if feat_class not in self.feature_to_idx:
        continue
      for feat, count in self.feature_statistics.feature_rep_dict[feat_class].items():
        if count >= self.threshold:
          self.feature_to_idx[feat_class][feat] = self.n_total_features
          self.n_total_features += 1
    print(f"you have {self.n_total_features} features!")

  def calc_represent_input_with_features(self) -> None:
    """
        initializes the matrices used in the optimization process - self.big_matrix and self.small_matrix
        """
    big_r = 0
    big_rows = []
    big_cols = []
    small_rows = []
    small_cols = []
    for small_r, hist in enumerate(self.feature_statistics.histories):
      for c in represent_input_with_features(hist, self.feature_to_idx):
        small_rows.append(small_r)
        small_cols.append(c)
      for r, y_tag in enumerate(self.feature_statistics.tags):
        demi_hist = (hist[0], y_tag, hist[2], hist[3], hist[4], hist[5], hist[6])
        self.histories_features[demi_hist] = []
        for c in represent_input_with_features(demi_hist, self.feature_to_idx):
          big_rows.append(big_r)
          big_cols.append(c)
          self.histories_features[demi_hist].append(c)
        big_r += 1
    self.big_matrix = sparse.csr_matrix((np.ones(len(big_rows)), (np.array(big_rows), np.array(big_cols))),
                                        shape=(len(self.feature_statistics.tags) * len(
                                          self.feature_statistics.histories), self.n_total_features),
                                        dtype=bool)
    self.small_matrix = sparse.csr_matrix(
      (np.ones(len(small_rows)), (np.array(small_rows), np.array(small_cols))),
      shape=(len(
        self.feature_statistics.histories), self.n_total_features), dtype=bool)


def represent_input_with_features(history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple[str, str], int]]) \
  -> List[int]:
  """
        Extract feature vector in per a given history
        @param history: tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}
        @param dict_of_dicts: a dictionary of each feature and the index it was given
        @return a list with all features that are relevant to the given history
    """
  c_word = history[0]
  c_tag = history[1]
  prev_word = history[2]
  prev_tag = history[3]
  pp_tag = history[5]
  next_word = history[6]
  features = []
  number_like = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "tens",
                 "eleven", "twelve", "twenty", "fifty", "hundred", "hundreds", "thousand", "thousands"]

  # f100
  if (c_word, c_tag) in dict_of_dicts["f100"]:
    features.append(dict_of_dicts["f100"][(c_word, c_tag)])

  len_word = len(c_word)
  for i in range(1, 5):
    if len_word >= i:  # because no there is no purpose to suffix with a world of 3
      suf = c_word[len_word - i:]
      pref = c_word[:i]
      # f101
      if (suf, c_tag) in dict_of_dicts["f101"]:
        features.append(dict_of_dicts["f101"][(suf, c_tag)])
      # f102
      if (pref, c_tag) in dict_of_dicts["f102"]:
        features.append(dict_of_dicts["f102"][(pref, c_tag)])
  # f103
  if (pp_tag, prev_tag, c_tag) in dict_of_dicts["f103"]:
    features.append(dict_of_dicts["f103"][(pp_tag, prev_tag, c_tag)])
  # f104
  if (prev_tag, c_tag) in dict_of_dicts["f104"]:
    features.append(dict_of_dicts["f104"][(prev_tag, c_tag)])
  # f105
  if (c_tag,) in dict_of_dicts["f105"]:
    features.append(dict_of_dicts["f105"][(c_tag,)])
  # f106
  if (prev_word, c_tag) in dict_of_dicts["f106"]:
    features.append(dict_of_dicts["f106"][(prev_word, c_tag)])
  # f107
  if (next_word, c_tag) in dict_of_dicts["f107"]:
    features.append(dict_of_dicts["f107"][(next_word, c_tag)])
  # f108
  if any(char.isdigit() for char in c_word):
    if (c_word, c_tag) in dict_of_dicts["f108"]:
      features.append(dict_of_dicts["f108"][(c_word, c_tag)])
  # f109
  if any(char.isupper() for char in c_word):
    if (c_word, c_tag,) in dict_of_dicts["f109"]:
      features.append(dict_of_dicts["f109"][(c_word, c_tag)])
  # f110
  if all(char.isupper() for char in c_word):
    if (c_word, c_tag) in dict_of_dicts["f110"]:
      features.append(dict_of_dicts["f110"][(c_word, c_tag)])
  # f111
  if c_word in number_like:
    if (c_word, c_tag) in dict_of_dicts["f111"]:
      features.append(dict_of_dicts["f111"][(c_word, c_tag)])
  # f112
  if any(char.isupper() for char in c_word) and any(char.isdigit() for char in c_word) and any(
    char == '-' for char in c_word):
    if (c_word, c_tag) in dict_of_dicts["f112"]:
      features.append(dict_of_dicts["f112"][(c_word, c_tag)])
  # f113
  if '-' in c_word:
    if (c_word, c_tag) in dict_of_dicts["f113"]:
      features.append(dict_of_dicts["f113"][(c_word, c_tag)])
  # f114
  if c_word[-4:] == 'ness' and c_tag == 'NN':
    if (c_word, c_tag,) in dict_of_dicts["f114"]:
      features.append(dict_of_dicts["f114"][(c_word, c_tag,)])
  """# f115
  if prev_tag == 'NN' and c_tag == 'JJ':
    if (prev_tag, c_tag) in dict_of_dicts["f115"]:
      features.append(dict_of_dicts["f115"][(prev_tag, c_tag)])
  # f116
  if prev_tag == 'JJ' and c_tag == 'NN':
    if (prev_tag, c_tag) in dict_of_dicts["f116"]:
      features.append(dict_of_dicts["f116"][(prev_tag, c_tag)])"""
  """# f117
  if c_word[-5:] == 'kappa' and c_tag == 'NN':
    if (c_word, c_tag) in dict_of_dicts["f117"]:
      features.append(dict_of_dicts["f117"][(c_word, c_tag,)])
  # f118
  if c_word[-5:] == 'virus' and c_tag == 'NN':
    if (c_word, c_tag) in dict_of_dicts["f118"]:
      features.append(dict_of_dicts["f118"][(c_word, c_tag)])
  # f119
  if c_word[-2:] == 'ol' and c_tag == 'NN':
    if (c_word, c_tag) in dict_of_dicts["f119"]:
      features.append(dict_of_dicts["f119"][(c_word, c_tag)])
  # f120
  if c_word[-2:] == 'an' and c_tag == 'NN':
    if (c_word, c_tag) in dict_of_dicts["f120"]:
      features.append(dict_of_dicts["f120"][(c_word, c_tag)])"""

  return features


def preprocess_train(train_path, threshold):
  # Statistics
  statistics = FeatureStatistics()

  statistics.get_word_tag_pair_count(train_path)

  # feature2id
  feature2id = Feature2id(statistics, threshold)
  feature2id.get_features_idx()
  feature2id.calc_represent_input_with_features()
  print(feature2id.n_total_features)

  for dict_key in feature2id.feature_to_idx:
    print(dict_key, len(feature2id.feature_to_idx[dict_key]))
  return statistics, feature2id


def read_test(file_path, tagged=True) -> List[Tuple[List[str], List[str]]]:
  """
    reads a test file
    @param file_path: the path to the file
    @param tagged: whether the file is tagged (validation set) or not (test set)
    @return: a list of all the sentences, each sentence represented as tuple of list of the words and a list of tags
    """
  list_of_sentences = []
  with open(file_path) as f:
    for line in f:
      if line[-1:] == "\n":
        line = line[:-1]
      sentence = (["*", "*"], ["*", "*"])
      split_words = line.split(' ')
      for word_idx in range(len(split_words)):
        if tagged:
          cur_word, cur_tag = split_words[word_idx].split('_')
        else:
          cur_word, cur_tag = split_words[word_idx], ""
        sentence[WORD].append(cur_word)
        sentence[TAG].append(cur_tag)
      sentence[WORD].append("~")
      sentence[TAG].append("~")
      list_of_sentences.append(sentence)
  return list_of_sentences
