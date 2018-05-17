import tensorflow as tf


class RnnConfig(object):
    """
    Rnn configure
    """

    def __init__(self, num_class, dim, vocab_size, seq_length=600, num_lay=2, hidden_dim=128, rnn='lstm',
                 drop_keep_prob=0.5, learning_rate=1e-3, batch_size=128, num_epochs=50, print_per_batch=10,
                 save_per_batch=10, embedding='one hot'):
        """
       the initialization of rnn_config
       :param num_class: the number of categories
       :param dim: the word embedding dim
       :param vocab_size: the number of words in data
       :param seq_length: the number of char each document
       :param num_lay: the number of hidden layer
       :param hidden_dim: the dim of hidden layer
       :param rnn: the kind of rnn
       :param drop_keep_prob: the drop out value
       :param learning_rate: the learning rate value
       :param batch_size: the size of each batch
       :param num_epochs: the number of epochs
       :param print_per_batch: when output the result
       :param save_per_batch: when save the model
       :param embedding: the kind of word embedding
       """

        self.embedding_dim = dim
        self.seq_length = seq_length
        self.num_classes = num_class
        self.vocab_size = vocab_size

        self.num_layer = num_lay
        self.hidden_dim = hidden_dim
        self.rnn = rnn

        self.dropout_keep_prob = drop_keep_prob
        self.learning_rate = learning_rate

        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.print_per_batch = print_per_batch
        self.save_per_batch = save_per_batch

        self.embedding = embedding


class TextRnn(object):
    """
    Rnn network
    """

    def __init__(self, config):
        """
        initalizate rnn net work
        :param config:
        """
        self.config = config

        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.rnn()

    def rnn(self):
        """
        bulid up the rnn network
        :return:
        """

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout():
            if self.config.rnn == 'lstm':
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        with tf.device('/gpu:0'):
            if self.config.embedding == 'embedding':
                W = tf.Variable(tf.constant(0.0, shape=[self.config.vocab_size, self.config.embedding_dim]),
                                trainable=False, name='W')
                self.embedding_placeholder = tf.placeholder(tf.float32,
                                                            [self.config.vocab_size, self.config.embedding_dim])
                self.embedding_init = W.assign(self.embedding_placeholder)
                embedding_inputs = tf.nn.embedding_lookup(W, self.input_x)
                print('load the pretrained word vector')
            else:
                embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
                embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
                print('load the one hot')

        with tf.name_scope("rnn"):
            cells = [dropout() for _ in range(self.config.num_layer)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]

        with tf.name_scope("score"):
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope("optimize"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            correnct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correnct_pred, tf.float32))