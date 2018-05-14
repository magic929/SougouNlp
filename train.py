from datetime import timedelta
import numpy as np

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
        
def feed_data(x_batch, y_batch, keep_prob):
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

def evaluate(sess, x_, y_):
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
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len

def train(model, config):
    """
    train the model
    :param model -> tf.graph: network  
    :param config -> class: RnnConfig class 
    :return: 
    """
    print("Configuring TensorBoard and Saver...")
    tensorboard_dir = './cnews/TextRnn'

    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    saver = tf.train.Saver()

    print('Loading training and validation data...')
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    # x_train = x_train[:100]
    # y_train = y_train[:100]
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    # x_val = x_val[:50]
    # y_val = y_val[:50]
    time_dif = get_time_dif(start_time)
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
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)
            # print(feed_dict)

            if total_batch % config.save_per_batch == 0:
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo evaluate

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
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