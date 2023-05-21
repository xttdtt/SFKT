import os
import tensorflow as tf


# calculate the total number of parameters
def calculate_parameter(savefile):
    checkpoint_path = os.path.join(dataset, 'Model') + '/' + savefile + '.ckpt'
    model_reader = tf.train.NewCheckpointReader(checkpoint_path)
    para_dict = model_reader.get_variable_to_shape_map()
    total_parameters = 0
    for key in para_dict:
        key_shape = np.shape(model_reader.get_tensor(key))
        key_parameters = 1
        for dim in key_shape:
            key_parameters *= dim
        # print("-------------------------------------------")
        # print("key variable name:", key)
        # print("key variable shape:", key_shape)
        # print("key variable parameters:", key_parameters)
        # print("key variable:\n", model_reader.get_tensor(key))
        total_parameters += key_parameters
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
embed_dim = 128
