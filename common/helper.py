import fasttext
import tensorflow.contrib.keras as kr

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


    


