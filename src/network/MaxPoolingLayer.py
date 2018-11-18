import tensorflow as tf

class MaxPoolingLayer(object):

    @staticmethod
    def feed_input(input_tensor, stride):
        _, shape_x, _ = input_tensor.shape
        return tf.map_fn(lambda inp: tf.convert_to_tensor([[tf.reduce_max(MaxPoolingLayer
                .__get_input_slice__(inp, x, y, stride))
            for x in range(0, shape_x - stride + 1, stride)]
            for y in range(0, shape_x - stride + 1, stride)], tf.float32), input_tensor)

    @staticmethod
    def __get_input_slice__(input_tensor, begin_x, begin_y, stride_width):
        return tf.slice(input_tensor, [begin_y, begin_x], [stride_width, stride_width])