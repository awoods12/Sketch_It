import tensorflow as tf
from src.network.Model import Model
from PIL import Image
import numpy as np
import os

#
# This is an example script for reading in files and labels and running a training process.
# Input in this case is any image file, labels are read in as [n,28*28] shape numpy bitmaps where n is the number of
# bitmaps per label.
#
# Input and label directory/file structure for this example is:
# One directory that contains directories for every image type with its label  as the directory name
# (e.g. ...\images\truck\this_is_a_truck.jpeg)
# One directory that contains numpy bitmaps whose file names map to your label directory names
# (e.g.) ...\labels\truck.npy)
#

open_image_dir = '...\\open_images\\' # replace with your path
labels = os.listdir(open_image_dir)

sess = tf.Session()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.01)

# input size is n where input images are reshaped to n*n using PIL
input_size = 20
input_matrix = tf.placeholder(tf.float32, shape=[1,input_size,input_size])
label_ph = tf.placeholder(tf.float32, shape=[28*28])

# takes input of shape [m,n,n], m is 1 for grayscale, m > 1 will apply 2d ([filter_size, filter_size]) conv filters on
# each n*n, not 3d conv filters like is often done for RGB (e.g. [3, filter_size, filter_size] filter shape for
# each filter)
model_out = Model.feed_forward(input_matrix)

# if evaluating or further training pre-trained weights, initialize a saver and run saver.restore instead of this init
init = tf.global_variables_initializer()
sess.run(init)

loss = tf.losses.mean_squared_error([tf.convert_to_tensor(label_ph, tf.float32)], [model_out])
train = optimizer.minimize(loss)

for i, label in labels:
    label_path = '...\\numpy_bitmap\\' + label.lower() + '.npy'  # replace with your path
    bitmaps = np.load(label_path)
    for bitmap in bitmaps:
        label_image = [val/256 for val in bitmap]

        curr_open_dir = open_image_dir + label
        open_images = os.listdir(curr_open_dir)

        for open_image in open_images:
            img = Image.open(curr_open_dir + '\\' + open_image).convert('L')
            img = img.resize((input_size, input_size), Image.ANTIALIAS)
            img = list(img.getdata())
            img = np.array([val/256 for val in img])
            img = img.reshape([input_size, input_size])
            img = img.reshape((1,) + img.shape)

            sess.run(train, feed_dict={input_matrix:img, label_ph:label_image})

saver = tf.train.Saver()
saver.save(sess,'...\\your_save_path')
