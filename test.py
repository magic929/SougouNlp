import time
import tensorflow as tf
import numpy as np
from sklearn import metrics
from datetime import timedelta
from optparse import OptionParser
import common.helper as helper
from model.rnn_model import *


def test(test_dir, config, model_dir):
    """
    test the documents 
    :param test_dir -> string: test data 
    :param config -> network class: network config  
    :param model_dir -> string: network model  
    :return: 
    """
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = helper.process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    tf.reset_default_graph()

    model = TextRnn(rnn_config)
    sess = tf.Session(config=gpu_config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path=model_dir)

    print('Testing...')
    loss_test, acc_test = helper.evaluate(sess, x_test, y_test, model)
    msg = 'Test Loss:{0:>6.2}, Test Acc:{1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = config.batch_size
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1
    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id: end_id],
            model.keep_prob: 1.0
        }

        y_pred_cls[start_id:end_id] = sess.run(model.y_pred_cls, feed_dict=feed_dict)

    print('Precision, Recall and F1-Socre...')
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories).encode('utf-8'))

    print('Confusion Matrix...')
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = helper.get_time_dif(start_time)
    print("Time usage:", time_dif)
    

def test_one(documemt, model_dir, config):
    """
    predict category of one document
    :param documemt -> string: document 
    :param model_dir ->: path of network model 
    :param config -> : config of network
    :return: 
    """
    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=gpu_config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path=model_dir)

    data_id = [[word_to_id[x] for x in documemt if x in word_to_id]]
    # print(data_id)
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, config.seq_length, padding='post', truncating='post')

    y_pred_cls = np.zeros(shape=1, dtype=np.int32)
    feed_dict = {
        model.input_x: x_pad,
        model.keep_prob: 1.0
    }

    y_pred_cls[0] = sess.run(model.y_pred_cls, feed_dict=feed_dict)
    for category in cat_to_id.keys():
        if cat_to_id[category] == y_pred_cls[0]:
            print(category)

            
if __name__ == '__main__':
    parser = OptionParser()
    (options, args) = parser.parse_args()
    test_dir = args[0]
    model_dir = args[1]
    vocab_dir = args[2]
    categories = ['体育' ,'财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    _, cat_to_id  = helper.read_category(categories)
    words, word_to_id = helper.read_vocab(vocab_dir)
    rnn_config = RnnConfig(len(categories), 64, len(words), rnn='gru', drop_keep_prob=0.8)
    test(test_dir, rnn_config, model_dir)
            
