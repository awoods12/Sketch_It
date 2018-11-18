from src.network.ConvLayer import ConvLayer
from src.network.MaxPoolingLayer import MaxPoolingLayer
from src.network.FullyConnectedLayer import FullyConnectedLayer
import tensorflow as tf

class Model(object):

    @staticmethod
    def feed_forward(input_matrix):
        input_matrix = tf.convert_to_tensor(input_matrix, dtype=tf.float32)

        first_conv_layer = ConvLayer.inception_layer('conv_one', 2, input_matrix, 1)

        first_conv_activated = tf.sigmoid(first_conv_layer)

        second_conv_layer = ConvLayer.inception_layer('conv_two', 2, first_conv_activated, 1)

        second_conv_activated = tf.sigmoid(second_conv_layer)

        first_max_pooling = MaxPoolingLayer.feed_input(second_conv_activated, 2)

        shape_1, shape_2, shape_3 = first_max_pooling.shape
        first_fully_layer = tf.reshape(first_max_pooling, [shape_1 * shape_2 * shape_3])

        first_weights = FullyConnectedLayer.xavier_weight_init('first_weights', int(first_fully_layer.shape[0]), 28*28)

        final_out = tf.nn.relu(FullyConnectedLayer.feed_input(first_fully_layer, first_weights,
                                                             FullyConnectedLayer.bias_init('bias_2')))

        return final_out