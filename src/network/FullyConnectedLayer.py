import tensorflow as tf

class FullyConnectedLayer(object):

    @staticmethod
    def xavier_weight_init(name, input_size, output_size):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            return [tf.get_variable(name=name + str(x), initializer=(tf.random_normal([input_size],
                    stddev=tf.sqrt(2 / (input_size + output_size)), mean=0, dtype=tf.float32)))
                for x in range(output_size)]

    @staticmethod
    def bias_init(name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            return tf.get_variable(name=name, initializer=0.0)

    @staticmethod
    def feed_input(input_tensor, weights, bias):
        return tf.convert_to_tensor([tf.add(tf.tensordot(input_tensor, w, 1), bias)
                for w in weights], dtype=tf.float32)