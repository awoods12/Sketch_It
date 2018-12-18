# Sketch_It
Convolutional neural network made with Tensorflow. My purpose is to train it to output crude drawings of given images by leveraging the Google quickdraw and Google open images datasets.

Current training progress: Training is underway, still working out the kinks. 

The goal for this project is to relate images to drawings of the same things and see if a convolutional network can pick up patterns between the images and drawings. If it works the network will output an image that looks something like a drawing of the input image (a photo of something simple like a truck or a bicycle). 

The project also includes just some general convolutional network building blocks (for tensorflow) that can be used to create networks that take completely different input and give different output.

The current model in model.py has layers conv_1->conv_2->max_pool->fully_connected->output. The functions available in the layer classes make it easy to redesign and expand this model's layers however you want by changing model.py.

Example.py gives an example of how to train the network (just some tensorflow basics and simple image processing). Take a look at it to get an idea of how you can use this package. 

The data I am using is all publicly available from the following locations:
QuickDraw: https://github.com/googlecreativelab/quickdraw-dataset
Open Images(Google Bigquery): https://console.cloud.google.com/bigquery?p=bigquery-public-data

![alt text](https://github.com/awoods12/Sketch_It/blob/master/images/converted_312524106_4ac9aa862b_o.jpg)
![alt text](https://github.com/awoods12/Sketch_It/blob/master/images/Vanconverted_6763235_df4352292b_o_50.jpg)
