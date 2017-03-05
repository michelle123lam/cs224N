import tensorflow as tf

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.

    TODO: Fill in what n_window_features should be, using n_word_features and window_size.
    """
    n_word_features = 2 # Number of features for every word in the input.
    window_size = 1 # The size of the window to use.
    ### YOUR CODE HERE
    n_window_features = n_word_features * (2 * window_size + 1)# The total number of features used for each window.
    ### END YOUR CODE
    n_classes = 5
    dropout = 0.5
    embed_size = 50
    hidden_size = 200
    batch_size = 2048
    n_epochs = 10
    lr = 0.001

    def __init__(self, output_path=None):
        if output_path:
            # Where to save things.
            self.output_path = output_path
        else:
            self.output_path = "results/window/{:%Y%m%d_%H%M%S}/".format(datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.log_output = self.output_path + "log"
        self.conll_output = self.output_path + "window_predictions.conll"

class WindowModel(NERModel):
    """
    Implements a feedforward neural network with an embedding layer and
    single hidden layer.
    This network will predict what label (e.g. PER) should be given to a
    given token (e.g. Manning) by  using a featurized window around the token.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, n_window_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None,), type tf.int32
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.dropout_placeholder

        (Don't change the variable names)
        """
        ### YOUR CODE HERE (~3-5 lines)
        self.input_placeholder = tf.placeholder(tf.int32, [None, self.config.n_window_features])
        self.labels_placeholder = tf.placeholder(tf.int32, [None,])
        self.dropout_placeholder = tf.placeholder(tf.float32, None)
        ### END YOUR CODE

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the model.
        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Hint: When an argument is None, don't add it to the feed_dict.

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE HERE (~5-10 lines)
        if inputs_batch is not None and labels_batch is not None and dropout is not None:
            feed_dict = {
                self.input_placeholder: inputs_batch,
                self.labels_placeholder: labels_batch,
                self.dropout_placeholder: dropout
            }
        elif inputs_batch is not None and labels_batch is None and dropout is not None:
            feed_dict = {
                self.input_placeholder: inputs_batch,
                self.dropout_placeholder: dropout
        }
        ### END YOUR CODE
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
            - Creates an embedding tensor and initializes it with self.pretrained_embeddings.
            - Uses the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, n_window_features, embedding_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, n_window_features * embedding_size).

        Hint: You might find tf.nn.embedding_lookup useful.
        Hint: You can use tf.reshape to concatenate the vectors. See following link to understand
            what -1 in a shape means.
            https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape.
        Returns:
            embeddings: tf.Tensor of shape (None, n_window_features*embed_size)
        """
        ### YOUR CODE HERE (!3-5 lines)

        embedding = tf.Variable(tf.convert_to_tensor(self.pretrained_embeddings))
        embedding_lookup_result = tf.nn.embedding_lookup(embedding, self.input_placeholder)
        embeddings = tf.reshape(embedding_lookup_result, [-1, self.config.n_window_features * self.config.embed_size])

        ### END YOUR CODE
        return embeddings

    def add_prediction_op(self):
        """Adds the 1-hidden-layer NN:
            h = Relu(xW + b1)
            h_drop = Dropout(h, dropout_rate)
            pred = h_dropU + b2

        Recall that we are not applying a softmax to pred. The softmax will instead be done in
        the add_loss_op function, which improves efficiency because we can use
        tf.nn.softmax_cross_entropy_with_logits

        When creating a new variable, use the tf.get_variable function
        because it lets us specify an initializer.

        Use tf.contrib.layers.xavier_initializer to initialize matrices.
        This is TensorFlow's implementation of the Xavier initialization
        trick we used in last assignment.

        Note: tf.nn.dropout takes the keep probability (1 - p_drop) as an argument.
            The keep probability should be set to the value of dropout_rate.

        Returns:
            pred: tf.Tensor of shape (batch_size, n_classes)
        """

        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder
        ### YOUR CODE HERE (~10-20 lines)
        xavier_initializer = tf.contrib.layers.xavier_initializer()

        W_shape =[self.config.n_window_features * self.config.embed_size, self.config.hidden_size]
        b1_shape = [self.config.hidden_size,]
        U_shape = [self.config.hidden_size, self.config.n_classes]
        b2_shape = [self.config.n_classes,]
        W = tf.get_variable("W", initializer=xavier_initializer(W_shape))
        b1 = tf.get_variable("b1", initializer=tf.zeros(b1_shape))
        U = tf.get_variable("U", initializer=xavier_initializer(U_shape))
        b2 = tf.get_variable("b2", initializer=tf.zeros(b2_shape))

        #compute
        h = tf.nn.relu(tf.matmul(x, W) + b1)
        h_drop = tf.nn.dropout(h, self.dropout_placeholder)
        pred = tf.matmul(h_drop, U) + b2
        ### END YOUR CODE
        return pred

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        In this case we are using cross entropy loss.
        The loss should be averaged over all examples in the current minibatch.

        Remember that you can use tf.nn.sparse_softmax_cross_entropy_with_logits to simplify your
        implementation. You might find tf.reduce_mean useful.
        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE (~2-5 lines)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(pred, self.labels_placeholder)
        loss = tf.reduce_mean(losses)
        ### END YOUR CODE
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        ### YOUR CODE HERE (~1-2 lines)
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        ### END YOUR CODE
        return train_op

    def preprocess_sequence_data(self, examples):
        return make_windowed_data(examples, start=self.helper.START, end=self.helper.END, window_size=self.config.window_size)

    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        ret = []
        #pdb.set_trace()
        i = 0
        for sentence, labels in examples_raw:
            labels_ = preds[i:i+len(sentence)]
            i += len(sentence)
            ret.append([sentence, labels, labels_])
        return ret

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(tf.argmax(self.pred, axis=1), feed_dict=feed)
        return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def __init__(self, helper, config, pretrained_embeddings, report=None):
        super(WindowModel, self).__init__(helper, config, report)
        self.pretrained_embeddings = pretrained_embeddings

        # Defining placeholders.
        self.input_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None

        self.build()


