import tensorflow as tf


class ConvNet:
    
    def __init__(self, params, trainable):
        self.shape = [None, params.width, params.height, params.history_length]
        self.x = tf.placeholder(tf.float32, self.shape)
        self.in_dims = self.shape[1]*self.shape[2]*self.shape[3]
        self.out_dims = params.actions
        self.filters = [32, 64, 64] # convolution filters at each layer
        self.num_layers = 3 # number of convolutional layers
        self.filter_size = [8, 4, 4] # size at each layer
        self.filter_stride = [4, 2, 1] # stride at each layer
        self.fc_size = [512] # size of fully connected layers
        self.fc_layers = 1 # number of fully connected layers
        self.trainable = trainable

        # dictionary for weights in network
        self.weights = {}
        # get predicted activation
        self.y = self.infer(self.x)

    def create_weight(self, shape):
        init = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(init, name='weight')

    def create_bias(self, shape):
        init = tf.constant(0.01, shape=shape)
        return tf.Variable(init, name='bias')

    def create_conv2d(self, x, w, stride):
        return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')

    def max_pool(self, x, size):
        return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')

    def infer(self, _input):
        self.layers = [_input]

        # initialize convolution layers
        for layer in range(self.num_layers):
            with tf.variable_scope('conv' + str(layer)) as scope:
                if layer == 0:
                    in_channels = self.shape[-1]
                    out_channels = self.filters[layer]
                else:
                    in_channels = self.filters[layer-1]
                    out_channels = self.filters[layer]

                shape = [ self.filter_size[layer], 
                          self.filter_size[layer],
                          in_channels, 
                          out_channels ]

                w = self.create_weight(shape)
                conv = self.create_conv2d(self.layers[-1], w, self.filter_stride[layer])

                b = self.create_bias([out_channels])
                self.weights[w.name] = w
                self.weights[b.name] = b
                bias = tf.nn.bias_add(conv, b)
                conv = tf.nn.relu(bias, name=scope.name)
                self.layers.append(conv)

        last_conv = self.layers[-1]

        # flatten last convolution layer
        dim = 1
        for d in last_conv.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(last_conv, [-1, dim], name='flat')
        self.layers.append(reshape)

        # initialize fully-connected layers
        for layer in range(self.fc_layers):
            with tf.variable_scope('hidden' + str(layer)) as scope:
                if layer == 0:
                    in_size = dim
                else:
                    in_size = self.fc_size[layer-1]

                out_size = self.fc_size[layer]
                shape = [in_size, out_size]
                w = self.create_weight(shape)
                b = self.create_bias([out_size])
                self.weights[w.name] = w
                self.weights[b.name] = b
                hidden = tf.nn.relu_layer(self.layers[-1], w, b, name=scope.name)
                self.layers.append(hidden)

        # create last fully-connected layer
        with tf.variable_scope('output') as scope:
            in_size = self.fc_size[self.fc_layers - 1]
            out_size = self.out_dims
            shape = [in_size, out_size]
            w = self.create_weight(shape)
            b = self.create_bias([out_size])
            self.weights[w.name] = w
            self.weights[b.name] = b
            hidden = tf.nn.bias_add(tf.matmul(self.layers[-1], w), b)
            self.layers.append(hidden)

        # return activation of the network
        return self.layers[-1]
