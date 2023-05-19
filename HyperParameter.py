import os
import tensorflow as tf


# calculate the total number of parameters
def calculate_parameter(savefile):
    modelfolder = os.path.join(dataset, 'Model')
    tf.train.import_meta_graph(modelfolder + '/' + savefile + '.ckpt.meta')
    variables = tf.trainable_variables()
    total_parameters = 0
    for variable in variables:
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


"""Here are some fixed parameters"""
# the maximum number of program training
epochs = 1000
# batch size of samples
bs = 256
# early-stop mechanism
early_stop = 20
# dropout ratio
keep_rate = 0.5
# ratio of training set and test set
split_rate = 0.7

"""
Here are some non-fixed parameters
When performing ablation experiments, you can modify the values here
"""
# this program only supports two datasets:Assist09 and Assist12
dataset = "Assist09"
# learning rate
lr = 0.01
# dimensions of embedding matrix
embed_dim = 512
