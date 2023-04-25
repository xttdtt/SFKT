import datetime
import logging
import os
import shutil
import sys
import time
import tensorflow as tf
import numpy as np
import math
from scipy import sparse
from HyperParameter import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def cosine_similarity(num1, num2):
    num1 = tf.cast(num1, tf.float32)
    num2 = tf.transpose(tf.cast(num2, tf.float32))
    inner = tf.matmul(num1, num2)
    norm1 = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(num1), axis=1)), (-1, 1))
    norm2 = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(num2), axis=0)), (1, -1))
    norm = tf.matmul(norm1, norm2)
    cosine = inner / norm
    return cosine


starttime = time.time()
datafolder = os.path.join(dataset, 'Data')
modelfolder = os.path.join(dataset, 'Model')

if os.path.exists(modelfolder):
    # delete old folder and create a new folder
    shutil.rmtree(modelfolder)
    os.mkdir(modelfolder)
else:
    os.mkdir(modelfolder)

# load related data
pro_skill_true = sparse.load_npz(os.path.join(dataset, 'Data/pro_skill_sparse.npz'))
pro_pro_true = sparse.load_npz(os.path.join(dataset, 'Data/pro_pro_sparse.npz'))
skill_skill_true = sparse.load_npz(os.path.join(dataset, 'Data/skill_skill_sparse.npz'))
pro_diff_true = sparse.load_npz(os.path.join(dataset, 'Data/pro_diff_sparse.npz'))
skill_diff_true = sparse.load_npz(os.path.join(dataset, 'Data/skill_diff_sparse.npz'))
stu_skill_true = sparse.load_npz(os.path.join(dataset, 'Data/stu_skill_sparse.npz'))

# convert related data to arrays
pro_skill_dense = pro_skill_true.toarray()
pro_pro_dense = pro_pro_true.toarray()
skill_skill_dense = skill_skill_true.toarray()
pro_diff_dense = pro_diff_true.toarray()
skill_diff_dense = skill_diff_true.toarray()
stu_skill_dense = stu_skill_true.toarray()

[num_pro, num_skill], num_stu = pro_skill_true.shape, stu_skill_true.shape[0]

tf_pro = tf.placeholder(tf.int32, [None])
tf_pro_skill_targets = tf.placeholder(tf.float32, [None, num_skill], name='tf_pro_skill')
tf_pro_pro_targets = tf.placeholder(tf.float32, [None, num_pro], name='tf_pro_pro')
tf_skill_skill_targets = tf.placeholder(tf.float32, [num_skill, num_skill], name='tf_skill_skill')
tf_pro_diff_target = tf.placeholder(tf.float32, [1, None], name='tf_pro_diff')
tf_skill_diff_target = tf.placeholder(tf.float32, [1, num_skill], name='tf_skill_diff')
tf_stu_skill_target = tf.placeholder(tf.float32, [num_stu, num_skill], name='tf_stu_skill')

# problem data embedding matrix
pro_data_embedding_matrix = tf.get_variable('pro_data_embed_matrix', [num_pro, embed_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
# skill embedding matrix
skill_embedding_matrix = tf.get_variable('skill_embed_matrix', [num_skill, embed_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
# student embedding matrix
stu_embedding_matrix = tf.get_variable('stu_embed_matrix', [num_stu, embed_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
# problem difficulty embedding matrix
pro_diff_embedding_matrix = tf.get_variable('pro_diff_embed_matrix', [num_pro, embed_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
# skill difficulty embedding matrix
skill_diff_embedding_matrix = tf.get_variable('skill_diff_embed_matrix', [num_skill, embed_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
# calculate skill embedding after adding weights
skill_embedding_matrix = tf.multiply(skill_embedding_matrix, tf.nn.softmax(skill_embedding_matrix, axis=0))
# calculate problem-related skill embedding matrix
pro_skill_embedding_matrix = tf.matmul(pro_skill_dense, skill_embedding_matrix)
# calculate problem embedding matrix
pro_data_sum = tf.reshape(tf.reduce_sum(pro_data_embedding_matrix, 1), (-1, 1))
pro_skill_sum = tf.reshape(tf.reduce_sum(pro_skill_embedding_matrix, 1), (-1, 1))
pro_rate = tf.exp(pro_data_sum) / (tf.exp(pro_data_sum) + tf.exp(pro_skill_sum))
pro_embedding_matrix = pro_rate * pro_data_embedding_matrix + (1 - pro_rate) * pro_skill_embedding_matrix

pro_embed = tf.nn.embedding_lookup(pro_embedding_matrix, tf_pro)
pro_diff_embed = tf.nn.embedding_lookup(pro_diff_embedding_matrix, tf_pro)

# optimization of problem-skill relationships
tf_pro_skill_logits = cosine_similarity(pro_embed, skill_embedding_matrix)
tf_pro_skill_logits = tf.reshape(tf_pro_skill_logits, [-1])
tf_pro_skill_labels = tf.reshape(tf_pro_skill_targets, [-1])
mse_pro_skill = tf.reduce_mean(tf.square(tf_pro_skill_labels - tf_pro_skill_logits))

# optimization of problem-problem relationships
tf_pro_pro_logits = cosine_similarity(pro_embed, pro_embedding_matrix)
tf_pro_pro_logits = tf.reshape(tf_pro_pro_logits, [-1])
tf_pro_pro_labels = tf.reshape(tf_pro_pro_targets, [-1])
mse_pro_pro = tf.reduce_mean(tf.square(tf_pro_pro_labels - tf_pro_pro_logits))

# optimization of skill-skill relationships
tf_skill_skill_logits = cosine_similarity(skill_embedding_matrix, skill_embedding_matrix)
tf_skill_skill_logits = tf.reshape(tf_skill_skill_logits, [-1])
tf_skill_skill_labels = tf.reshape(tf_skill_skill_targets, [-1])
mse_skill_skill = tf.reduce_mean(tf.square(tf_skill_skill_labels - tf_skill_skill_logits))

# optimization of problem difficulty
tf_pro_diff_logits = tf.diag_part(tf.matmul(pro_embed, tf.transpose(pro_diff_embed)))
tf_pro_diff_logits = tf.reshape(tf_pro_diff_logits, [-1])
tf_pro_diff_labels = tf.reshape(tf_pro_diff_target, [-1])
mse_pro_diff = tf.reduce_mean(tf.square(tf_pro_diff_logits - tf_pro_diff_labels))

# optimization of skill difficulty
tf_skill_diff_logits = tf.diag_part(tf.matmul(skill_embedding_matrix, tf.transpose(skill_diff_embedding_matrix)))
tf_skill_diff_logits = tf.reshape(tf_skill_diff_logits, [-1])
tf_skill_diff_labels = tf.reshape(tf_skill_diff_target, [-1])
mse_skill_diff = tf.reduce_mean(tf.square(tf_skill_diff_logits - tf_skill_diff_labels))

# optimization of student's mastery degree of skills
tf_stu_skill_logits = tf.matmul(stu_embedding_matrix, tf.transpose(skill_embedding_matrix))
tf_stu_skill_logits = tf.reshape(tf_stu_skill_logits, [-1])
tf_stu_skill_labels = tf.reshape(tf_stu_skill_target, [-1])
mse_stu_skill = tf.reduce_mean(tf.square(tf_stu_skill_labels - tf_stu_skill_logits))

# overall optimization
loss = mse_pro_skill + mse_pro_pro + mse_skill_skill + mse_pro_diff + mse_skill_diff + mse_stu_skill
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)

# set log file to store the running results
logfile = os.path.join(modelfolder, "trainEmbedding.txt")
# clear log file content before each run
f = open(logfile, "wb+")
f.truncate()
logging.basicConfig(filename=logfile, level="DEBUG")
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

startTraintime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logging.info(os.linesep + '-' * 45 + ' BEGIN: ' + startTraintime + ' ' + '-' * 45)
logging.info('dataset %s, problem number %d, skill number %d, student number %d' % (dataset, num_pro, num_skill, num_stu))

train_steps = int(math.ceil(num_pro / float(bs)))

logging.info("begin training....")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    best_loss, tmp, Loss = np.inf, 0, np.zeros(epochs)
    for i in range(epochs):
        epochstarttime = time.time()
        train_loss = 0
        for m in range(train_steps):
            b, e = m * bs, min((m + 1) * bs, num_pro)
            batch_pro = np.arange(b, e).astype(np.int32)
            batch_pro_pro = pro_pro_dense[b:e:]
            batch_pro_skill = pro_skill_dense[b:e:]
            batch_pro_diff = pro_diff_dense[:, b:e]
            feed_dict = {tf_pro: batch_pro,
                         tf_pro_skill_targets: batch_pro_skill,
                         tf_pro_pro_targets: batch_pro_pro,
                         tf_skill_skill_targets: skill_skill_dense,
                         tf_pro_diff_target: batch_pro_diff,
                         tf_skill_diff_target: skill_diff_dense,
                         tf_stu_skill_target: stu_skill_dense}
            _, loss_ = sess.run([train_op, loss], feed_dict=feed_dict)
            train_loss += loss_
        train_loss /= train_steps
        epochendtime = time.time()
        logging.info("epoch %d, loss %f, time %f" % (i + 1, train_loss, epochendtime - epochstarttime))
        Loss[i] = train_loss
        if i >= early_stop:
            if all(x <= y for x, y in zip(Loss[tmp:tmp + early_stop], Loss[tmp + 1:tmp + 1 + early_stop])):
                logging.info("Early stop at %d based on loss result." % (i + 1))
                break
            tmp += 1
    logging.info('finish training...')

    final_pro_pro_embed = tf.convert_to_tensor(tf.multiply(pro_embedding_matrix, pro_diff_embedding_matrix))
    final_pro_pro_embed = final_pro_pro_embed.eval()
    final_pro_embed = np.array(final_pro_pro_embed)

    final_skill_embed = tf.convert_to_tensor(skill_embedding_matrix)
    final_skill_embed = final_skill_embed.eval()
    final_skill_embed = np.array(final_skill_embed)

    final_stu_embed = tf.convert_to_tensor(stu_embedding_matrix)
    final_stu_embed = final_stu_embed.eval()
    final_stu_embed = np.array(final_stu_embed)

    np.savez(os.path.join(modelfolder, 'final_pro_embed.npz'), final_pro_embed=final_pro_embed)
    np.savez(os.path.join(modelfolder, 'final_skill_embed.npz'), final_skill_embed=final_skill_embed)
    np.savez(os.path.join(modelfolder, 'final_stu_embed.npz'), final_stu_embed=final_stu_embed)

endTraintime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logging.info(os.linesep + '-' * 45 + ' END: ' + endTraintime + ' ' + '-' * 45)

endtime = time.time()
logging.info("total time %f" % (endtime - starttime))
