import time
import numpy as np
import tensorflow as tf
import common.helper as helper
from optparse import OptionParser
from model.rnn_model import *


def train(train_dir, val_dir, vocab_dir, save_path, tensorboard_dir, config):
    """
    train the model
    :param model -> tf.graph: network  
    :param config -> class: RnnConfig class 
    :return: 
    """
    model = TextRnn(rnn_config)
    print("Configuring TensorBoard and Saver...")
    tensorboard_dir = './cnews/TextRnn'

    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    saver = tf.train.Saver()

    print('Loading training and validation data...')
    start_time = time.time()
    x_train, y_train = helper.process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    # x_train = x_train[:100]
    # y_train = y_train[:100]
    x_val, y_val = helper.process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    # x_val = x_val[:50]
    # y_val = y_val[:50]
    time_dif = helper.get_time_dif(start_time)
    # print(x_train[0], y_train[0])
    # print(x_val[0], y_val[0])
    # print('time usage:', time_dif)

    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    session = tf.Session(config=gpu_config)
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)
    if config.embedding == 'embedding':
        session.run(model.embedding_init, feed_dict={model.embedding_placeholder: embedding})

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0
    best_acc_val = 0.0
    last_improved = 0
    require_improvement = 10

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = helper.batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = helper.feed_data(x_batch, y_batch, config.dropout_keep_prob, model)
            # print(feed_dict)

            if total_batch % config.save_per_batch == 0:
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = helper.evaluate(session, x_val, y_val, model)  # todo evaluate

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = helper.get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%}, ' + ' val Lpss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)
            total_batch += 1

            # if total_batch - last_improved > require_improvement:
            #  print("No optimization for a long time, auto-stopping...")
            #  flag = True
            #  break
            # if flag:
            # break 

if __name__ == '__main__':
    parser = OptionParser()
    (options, args) = parser.parse_args()
    train_dir = args[0]
    val_dir = args[1]
    vocab_dir = args[2]
    model_dir = args[3]
    checkpoint_dir = args[4]
    categories = ['体育' ,'财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    _, cat_to_id  = helper.read_category(categories)
    words, word_to_id = helper.read_vocab(vocab_dir)
    rnn_config = RnnConfig(len(categories), 64, len(words), rnn='gru', drop_keep_prob=0.8, num_epochs=25)
    train(train_dir, val_dir, vocab_dir, model_dir, checkpoint_dir, rnn_config)
