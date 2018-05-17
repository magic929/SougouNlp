import fasttext
import time
from datetime import timedelta
import tensorflow.contrib.keras as kr
from datetime import timedelta
import numpy as np


class fasttextConfig(object):
    """
    fasttext model class
    """
    def __init__(self, dim=100, ws=5, min_count=3):
        """
        fasattect model initialize
        :param dim: word embedding dim
        :param ws: size of windows in document
        :param min_count: drop the lower number of word
        """
        self.dim = dim
        self.ws = ws
        self.min_count = min_count
       
    def fasttext_model(doc, model):
        """
        generate without label and save it
        generate the fasttext model
        :param doc -> string: data path  
        :param model -> string: fasttext model path
        :return: 
        """
        with open('./cnews/without.dat', 'w', encoding='utf8') as f:
            data = read_file(doc)
            for d, _ in data:
                f.write(d)
                f.flush()        
        fasttext.skipgram('./cnews/without.dat', model,dim=self.dim, ws=self.ws, min_count=self.min_count)
        

def loadWordEmbedding(model):
    """
    load the pretrained word embedding
    :param model:
    :return:
    """
    model = fasttext.load_model(model)
    vocab = ['unk']
    embd = [[0] * 100]
    for word in model.words:
        vocab.append(word)
        embd.append(model[word])
    print("loaded word embedding")
    return vocab, embd


def read_category(categories):
    """
    convert the categories to one hot
    :param categories -> string: news categories
    :return: categories, the one hot of categories
    """
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def read_file(filename):
    """
    read news data
    :param filename -> string: news data
    :return: label -> stirng: news category
    """
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            yield (line.split('\t')[1], line.split('\t')[0])

            
def process_file(content_dir, word_to_id, cat_to_id, max_length=1000, embedding='one hot'):
    """
    process document and label to the format that fit the input:x and input:y
    :param content_dir -> stirng: file dir 
    :param word_to_id -> dict: the index to word 
    :param cat_to_id -> dict: the index to category 
    :param max_length -> int: the length of each document 
    :param embedding -> stirng: the kind of embedding   
    :return: formatted x, formatted y
    """
    contents = read_file(content_dir)
    raw_data = []
    labels = []
    for content in contents:
        raw_data.append(content[0])
        labels.append(content[1])

    data_id, label_id = [], []
    if embedding == 'one hot':
        for i in range(len(raw_data)):
            data_id.append([word_to_id[x] for x in raw_data[i] if x in word_to_id])
            label_id.append(cat_to_id[labels[i]])
            # print(data_id[len(data_id) - 1])
            # print(label_id[len(data_id) - 1])
        print('data processed!')
    else:
        data_id = [[word_to_id[x] if x in word_to_id else word_to_id['unk'] for x in raw.split(' ')] for raw in
                   raw_data]
        print('wordvector successfully')
        label_id = [cat_to_id[label] for label in labels]
        print('generated label ont hot')

    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', truncating='post')
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))

    return x_pad, y_pad


def generate_vocab_dict(vocab, vocab_dir):
    """
    make vocab and save it
    :param vocab: the list of words
    :param vocab_dir: saving path
    :return: dict -> the index of words
    """
    word_to_id = dict(zip(vocab, range(len(vocab))))
    with open(vocab_dir, 'w', encoding='utf8') as f:
        for key in word_to_id.keys():
            f.write(key + '\t' + str(word_to_id[key]) + '\n')
            f.flush()
    return word_to_id


def read_vocab(vocab_dir):
    """
    read the vocab
    :param vocab_dir -> string: vocab path
    :return: words, word_to_id
    """
    with open(vocab_dir, 'r', encoding='utf8') as f:
        words = [_.strip() for _ in f.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def get_time_dif(start_time):
    """
    compute the running rime 
    :param start_time -> time: begin time 
    :return: running time 
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def batch_iter(x, y, batch_size=64):
    """
    generate shuffle batch 
    :param x -> list: document 
    :param y -> list: label
    :param batch_size -> int: size of batch  
    :return: tuple: (shuffle_document, shuffle_label)
    """
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
        
def feed_data(x_batch, y_batch, keep_prob, model):
    """
    feed the data to network
    :param x_batch -> list(array): one of document batch
    :param y_batch -> list(array): one of label batch
    :param keep_prob: drop out value
    :return: dict: feed_dict
    """
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }

    return feed_dict


def evaluate(sess, x_, y_, model):
    """
    compute the accuracy and loss value of prediction
    :param sess -> tf.session: tensorflow session 
    :param x_ -> list: test documents
    :param y_ -> list: test labels
    :return: float: the average of loss value , accuracy
    """
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0, model)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len



    


