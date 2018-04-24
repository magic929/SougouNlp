# -*- coding: UTF-8 -*-

import jieba
import sys
import getopt
import fasttext
from sklearn.model_selection import train_test_split




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


def fasttext_model(doc, model):
    fasttext.skipgram(doc, model,
                      dim=param.dim, ws=param.ws, min_count=param.min_count, t=param.t, thread=param.thread)


def write_to_file(file, data):
    with open(file, 'w', encoding='utf8') as f:
        for x, y in data:
            f.write(x + '\t' + y)
            f.flush()

        f.close()


def read_file(file):
    with open(file, 'r', encoding='utf8') as f:
        doc = f.readlines()

    for d in doc:
        text = d.split('\t')
        yield (text[0], text[1])


def make_label(file, rfile):
    doc = read_file(rfile)
    with open(file, 'w', encoding='utf8') as f:
        for d in doc:
            f.write(d[0] + '\n') 

def select(argv):
    try:
        opts, args = getopt.getopt(argv, 't:c:f:l:')
    except getopt.GetoptError:
        print('error')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-t':
            to_train_test(read_file(arg))
        elif opt == '-c':
            cut_sentence(arg, args[0])
        elif opt == '-f':
            fasttext_model(arg, args[0])
        elif opt == '-l':
            make_label(arg, args[0])


if __name__ == "__main__":
    select(sys.argv[1:])
