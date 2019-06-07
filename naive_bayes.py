# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
from collections import defaultdict
import re

class NaiveBayes(Classifier):
    u"""A naïve Bayes classifier."""

    def __init__(self):
        self.train_dict = {}
        self.prob_dict = {}
        self.cat_count = {}
        self.category = []

    def train(self, list):
        print("start")
        print("features: ", list[0].features())
        print("label: ", list[0].label)
        print("data: ", list[0].data)


        for l in list:
            if l.label not in self.category:  # get all labels that need to be caterized (default 2 labels)
                self.category.append(l.label)
            if l.label not in self.cat_count:  # count how many elements under each label
                self.cat_count[l.label] = 1
            else:
                self.cat_count[l.label] += 1

            count_dict = {}
            for element in l.features():
                if element not in count_dict:
                    count_dict[element] = 1
                else:
                    count_dict[element] += 1

            for element in l.features():
                if (l.label, element, count_dict[element]) in self.train_dict:  # build training dictionary
                    self.train_dict[(l.label, element, count_dict[element])] += 1
                else:
                    self.train_dict[(l.label, element, count_dict[element])] = 1
                    # self.train_dict[(l.label, 'UNK', 'UNK')] = 1

        for (g, x, y) in self.train_dict:
            if (x, y) not in self.prob_dict:  # build probability dictionary (for prob calculation)
                self.prob_dict[(x, y)] = self.train_dict[(g, x, y)]
            else:
                self.prob_dict[(x, y)] += self.train_dict[(g, x, y)]

                # print(train_dict)
                # print(prob_dict)

    def classify(self, word):
        self.prob_list = [1, 1, 0]
        count_dict = {}
        for element in word.features():
            if element not in count_dict:
                count_dict[element] = 1
            else:
                count_dict[element] += 1

        for element in word.features():
            for j, c in enumerate(self.category):
                if (c, element, count_dict[element]) in self.train_dict:
                    self.prob_list[j] = self.prob_list[j] * (
                        self.train_dict[(c, element, count_dict[element])] / self.prob_dict[
                            (element, count_dict[element])])
                else:
                    self.prob_list[j] *= 0.1
        if self.prob_list[0] > self.prob_list[1]:
            return self.category[0]
        else:
            return self.category[1]
