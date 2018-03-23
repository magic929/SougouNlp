# -*- coding: UTF-8 -*-

import jieba
import sys
import getopt
import progressbar
from sklearn.model_selection import train_test_split

import config.param as param


def to_train_test(data):
    train = []
    test = []
    for l in param.categories:
        tmp_data = [(x, y) for x, y in data if x == l]
        train_label, test_label = train_test_split(tmp_data, test_size=param.size)
        train += train_label
        test += test_label

    write_to_file('./PreData/train', train)
    write_to_file('./PreData/test', test)


def cut_sentence(doc, file):
    data = read_file(doc)
    f = open(file, 'w', encoding='utf8')
    for x, y in data:
        words = jieba.cut(y)
        f.write(' '.join(words))
        f.flush()


def write_to_file(file, data):
    with open(file, 'w', encoding='utf8') as f:
        for x, y in data:
            f.write(x + '\t' + y)
            f.flush()

        f.close()


def read_file(file):
    with open(file, 'r', encoding='utf8') as f:
        doc = f.readlines()

    return [(d.split('\t')[0], d.split('\t')[1]) for d in doc]


def select(argv):
    try:
        opts, args = getopt.getopt(argv, 't:c:')
    except getopt.GetoptError:
        print('error')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-t':
            to_train_test(read_file(arg))
        elif opt == '-c':
            cut_sentence(arg, args[0])


if __name__ == "__main__":
    select(sys.argv[1:])
