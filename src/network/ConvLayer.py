import tensorflow as tf

class ConvLayer(object):

    @staticmethod
    def xavier_filter_init(name, filter_size, num_filters):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            return [tf.get_variable(name=name + str(x), initializer=(tf.random_normal([filter_size, filter_size],
                            stddev=tf.sqrt(1/(filter_size*filter_size)), mean=0, dtype=tf.float32)))
                    for x in range(num_filters)]

    @staticmethod
    def conv_biases(name, filters):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            return tf.get_variable(name=name, initializer=tf.zeros(len(filters)))

    @staticmethod
    def feed_input(input_tensors, filters, biases, stride, zero_pad=True):
        if zero_pad:
            pad_size = int((int(filters[0].shape[0]) + 1)/2 - 1)
            input_tensors = tf.map_fn(lambda inp:
                tf.pad(inp, [[pad_size, pad_size], [pad_size, pad_size]]), input_tensors)

        _, shape_x, _ = input_tensors.shape

        unflattened_output = tf.map_fn(lambda input_tensor: tf.convert_to_tensor([[[tf.add(tf.tensordot(curr_filter,
            ConvLayer.__get_input_slice__(input_tensor, x, y, tf.shape(curr_filter)[0]), 2), biases[filter_ind])
         for x in range(0, shape_x - curr_filter.shape[0] + 1, stride)]
         for y in range(0, shape_x - curr_filter.shape[0] + 1, stride)]
         for filter_ind, curr_filter in enumerate(filters)], dtype=tf.float32), input_tensors)

        shape_1, shape_2, shape_3, shape_4 = unflattened_output.shape

        return tf.reshape(unflattened_output, [shape_1 * shape_2, shape_3, shape_4])

    @staticmethod
    def inception_layer(name, num_each_filter, input_matrix, stride):
        conv_filters_1 = ConvLayer.xavier_filter_init(name=name + '_conv_filters_1', filter_size=1,
                                                      num_filters=num_each_filter)
        conv_filters_3 = ConvLayer.xavier_filter_init(name=name + '_conv_filters_3', filter_size=3,
                                                      num_filters=num_each_filter)
        conv_filters_5 = ConvLayer.xavier_filter_init(name=name + '_conv_filters_5', filter_size=5,
                                                      num_filters=num_each_filter)

        conv_biases_1 = ConvLayer.conv_biases(name + '_conv_biases_1', conv_filters_1)
        conv_biases_3 = ConvLayer.conv_biases(name + '_conv_biases_3', conv_filters_3)
        conv_biases_5 = ConvLayer.conv_biases(name + '_conv_biases_5', conv_filters_5)

        return tf.concat([ConvLayer.feed_input(input_matrix, conv_filters_1, conv_biases_1, stride),
                          ConvLayer.feed_input(input_matrix, conv_filters_3, conv_biases_3, stride),
                          ConvLayer.feed_input(input_matrix, conv_filters_5, conv_biases_5, stride)],
                         0)

    @staticmethod
    def __get_input_slice__(input_tensor, begin_x, begin_y, filter_width):
        return tf.slice(input_tensor, [begin_y, begin_x], [filter_width, filter_width])
