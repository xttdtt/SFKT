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
        total_shape = list(key_shape)
        # print("variable name: ", key)
        # print("variable shape: ", key_shape)
        # print(model_reader.get_tensor(key))
        parameters = 1
        for dim in total_shape:
            parameters *= dim
        total_parameters += parameters
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
# learning rate:0.001 0.005 0.01
lr = 0.01
# dimensions of embedding matrix：128  256  512
embed_dim = 128
