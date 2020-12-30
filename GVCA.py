import tensorflow as tf
import os
import loader_class_euc as loader_class
import random
import numpy as np
import heapq
from tensorflow.python.ops import math_ops, array_ops, random_ops, nn_ops
import sys
import scipy

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
assert sys.argv[1] == '--d'
database = sys.argv[2]
rgb_features_dim = 3*2048
depth_features_dim = 110
train_data = loader_class.data_loader(database)

if database == 'UWA' or database == 'UWA30':
    class_num = 30
    train_data.read_train_action()
elif database == 'UCB':
    class_num = 11
    train_data.read_train_action()
elif database == 'DHA':
    class_num = 23
    train_data.read_train_action()

rgb_features_l_1 = tf.placeholder(tf.float32, shape=[None, rgb_features_dim])
depth_features_l_1 = tf.placeholder(tf.float32, shape=[None, depth_features_dim])
rgb_features_l_2 = tf.placeholder(tf.float32, shape=[None, rgb_features_dim])
depth_features_l_2 = tf.placeholder(tf.float32, shape=[None, depth_features_dim])
rgb_features_u = tf.placeholder(tf.float32, shape=[None, rgb_features_dim])
depth_features_u = tf.placeholder(tf.float32, shape=[None, depth_features_dim])
alpha = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)
lr_r = tf.placeholder(tf.float32)
lr_d = tf.placeholder(tf.float32)

train_num = train_data.sample_total_num

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

label = tf.placeholder(tf.float32, shape=[None, class_num])
label_loss = tf.placeholder(tf.float32, shape=[None, class_num])
label_test = tf.placeholder(tf.float32, shape=[None, class_num])


adj = tf.placeholder(tf.float32, shape=[None, None])
B = tf.placeholder(tf.float32, shape=[None, None])

subspace_rgb_dim = 200
subspace_depth_dim = 100
subspace_rd_dim = 100
hidden_size = 100
rgb_W2 = tf.Variable(xavier_init([subspace_rgb_dim, class_num]))
rgb_b2 = tf.Variable(tf.zeros(shape=[class_num]))
rgb_W1 = tf.Variable(xavier_init([rgb_features_dim, subspace_rgb_dim]))
rgb_b1 = tf.Variable(tf.zeros(shape=[subspace_rgb_dim]))

dep_W2 = tf.Variable(xavier_init([subspace_depth_dim, class_num]))
dep_b2 = tf.Variable(tf.zeros(shape=[class_num]))
dep_W1 = tf.Variable(xavier_init([depth_features_dim, subspace_depth_dim]))
dep_b1 = tf.Variable(tf.zeros(shape=[subspace_depth_dim]))

feature3_dim = subspace_rgb_dim + subspace_depth_dim
rd_W2 = tf.Variable(xavier_init([subspace_rd_dim, class_num]))
rd_b2 = tf.Variable(tf.zeros(shape=[class_num]))
rd_W1 = tf.Variable(xavier_init([feature3_dim, subspace_rd_dim]))
rd_b1 = tf.Variable(tf.zeros(shape=[subspace_rd_dim]))

# Leaky Relu
def leak_relu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

step = 1

# classifier_one_view RGB sigmoid depth no sigmoid
def represent_one_view(inputs, W1, b1):
    layer1 = leak_relu(tf.matmul(inputs, W1) + b1, 0.2)
    return layer1

def classifier_one_view(inputs, W1, b1):
    layer1 = tf.matmul(inputs, W1) + b1
    return layer1


def pre_sig(rgb_features_l_1, rgb_features_l_2, depth_features_l_1, depth_features_l_2, rgb_features_u, depth_features_u, database):
    
    if database != 'nyu':
        rgb_features_l_1 = tf.nn.sigmoid(rgb_features_l_1)
        rgb_features_l_2 = tf.nn.sigmoid(rgb_features_l_2)
        rgb_features_u = tf.nn.sigmoid(rgb_features_u)
    
    return rgb_features_l_1, rgb_features_l_2, depth_features_l_1, depth_features_l_2, rgb_features_u, depth_features_u

_rgb_features_l_1, _rgb_features_l_2, _depth_features_l_1, _depth_features_l_2, _rgb_features_u, _depth_features_u = pre_sig(rgb_features_l_1, rgb_features_l_2, depth_features_l_1, depth_features_l_2, rgb_features_u, depth_features_u, database)

epsilon = 1e-6

def BN_layer(feature, W, b, scale):
    feature_l1 = tf.matmul(feature, W)
    feature_batch_mean1_l1, feature_batch_var1_l1 = tf.nn.moments(feature_l1,[0])
    feature_z1_hat_l1 = (feature_l1 - feature_batch_mean1_l1) / tf.sqrt(feature_batch_var1_l1 + epsilon)
    feature_BN_l1 = scale * feature_z1_hat_l1 + b
    return feature_BN_l1

e1_dim = subspace_rgb_dim
e1_w1_BN = tf.Variable(xavier_init([e1_dim, e1_dim]))
e1_scale1 = tf.Variable(tf.ones([e1_dim]))
e1_beta1 = tf.Variable(tf.zeros([e1_dim]))

e1_l1 = represent_one_view(_rgb_features_l_1, rgb_W1, rgb_b1)
e1_l2 = represent_one_view(_rgb_features_l_2, rgb_W1, rgb_b1)
e1_l = alpha * e1_l1 + (1 - alpha) * e1_l2
r1_l = classifier_one_view(e1_l, rgb_W2, rgb_b2)

e1_u = represent_one_view(_rgb_features_u, rgb_W1, rgb_b1)
r1_u = classifier_one_view(e1_u, rgb_W2, rgb_b2)

_rgb_features_l_1_BN = BN_layer(e1_l1, e1_w1_BN, e1_beta1, e1_scale1)
_rgb_features_l_2_BN = BN_layer(e1_l2, e1_w1_BN, e1_beta1, e1_scale1)
_rgb_features_u_BN = BN_layer(e1_u, e1_w1_BN, e1_beta1, e1_scale1)

e2_dim = subspace_depth_dim
e2_w1_BN = tf.Variable(xavier_init([e2_dim, e2_dim]))
e2_scale1 = tf.Variable(tf.ones([e2_dim]))
e2_beta1 = tf.Variable(tf.zeros([e2_dim]))

e2_l1 = represent_one_view(_depth_features_l_1, dep_W1, dep_b1)
e2_l2 = represent_one_view(_depth_features_l_2, dep_W1, dep_b1)
e2_l = alpha * e2_l1 + (1 - alpha) * e2_l2
r2_l = classifier_one_view(e2_l, dep_W2, dep_b2)

e2_u = represent_one_view(_depth_features_u, dep_W1, dep_b1)
r2_u = classifier_one_view(e2_u, dep_W2, dep_b2)

_depth_features_l_1_BN = BN_layer(e2_l1, e2_w1_BN, e2_beta1, e2_scale1)
_depth_features_l_2_BN = BN_layer(e2_l2, e2_w1_BN, e2_beta1, e2_scale1)
_depth_features_u_BN = BN_layer(e2_u, e2_w1_BN, e2_beta1, e2_scale1)

feature3_l1_0 = tf.concat([_rgb_features_l_1_BN, _depth_features_l_1_BN], axis=1)
feature3_l2_0 = tf.concat([_rgb_features_l_2_BN, _depth_features_l_2_BN], axis=1)
feature3_u_0 = tf.concat([_rgb_features_u_BN, _depth_features_u_BN], axis=1)

feature3_BN_l1 = feature3_l1_0
feature3_BN_l2 = feature3_l2_0
feature3_BN_u = feature3_u_0

e3_l1 = represent_one_view(feature3_BN_l1, rd_W1, rd_b1)
e3_l2 = represent_one_view(feature3_BN_l2, rd_W1, rd_b1)
e3_l = alpha * e3_l1 + (1 - alpha) * e3_l2
r3_l = classifier_one_view(e3_l, rd_W2, rd_b2)

e3_u = represent_one_view(feature3_BN_u, rd_W1, rd_b1)
r3_u = classifier_one_view(e3_u, rd_W2, rd_b2)

e1_u_1 = (1 - alpha) * e1_l1 + alpha * e1_u
e1_u_2 = (1 - alpha) * e1_l2 + alpha * e1_u
r1_u_1 = classifier_one_view(e1_u_1, rgb_W2, rgb_b2)
r1_u_2 = classifier_one_view(e1_u_2, rgb_W2, rgb_b2)

e2_u_1 = (1 - alpha) * e2_l1 + alpha * e2_u
e2_u_2 = (1 - alpha) * e2_l2 + alpha * e2_u
r2_u_1 = classifier_one_view(e2_u_1, dep_W2, dep_b2)
r2_u_2 = classifier_one_view(e2_u_2, dep_W2, dep_b2)

e3_u_1 = (1 - alpha) * e3_l1 + alpha *  e3_u
e3_u_2 = (1 - alpha) * e3_l2 + alpha *  e3_u
r3_u_1 = classifier_one_view(e3_u_1, rd_W2, rd_b2)
r3_u_2 = classifier_one_view(e3_u_2, rd_W2, rd_b2)


loss_repre_1 = tf.reduce_mean(tf.square(label_loss - (r1_u_1 - r1_u_2)))
loss_repre_2 = tf.reduce_mean(tf.square(label_loss - (r2_u_1 - r2_u_2)))
loss_repre_3 = tf.reduce_mean(tf.square(label_loss - (r3_u_1 - r3_u_2)))

def Entropy(p):
    p = tf.nn.softmax(p)
    log_p = tf.log(p)
    b = tf.ones(shape=[100, 1])
    return  tf.reduce_sum((- tf.multiply(p, log_p)), axis=1)

def sort_rows(matrix, num_rows):
    matrix_T = array_ops.transpose(matrix, [1, 0])
    sorted_matrix_T = nn_ops.top_k(matrix_T, num_rows)[0]
    return array_ops.transpose(sorted_matrix_T, [1, 0])

def discrepancy_slice_wasserstein(p1, p2):
    p1 = tf.nn.sigmoid(p1)
    p2 = tf.nn.sigmoid(p2)
    s = array_ops.shape(p1)
    if p1.get_shape().as_list()[1] > 1:
        # For data more than one-dimensional, perform multiple random projection to 1-D
        proj = random_ops.random_normal([array_ops.shape(p1)[1], 128])
        proj *= math_ops.rsqrt(math_ops.reduce_sum(math_ops.square(proj), 0, keepdims=True))
        p1 = math_ops.matmul(p1, proj)
        p2 = math_ops.matmul(p2, proj)
    p1 = sort_rows(p1, s[0])
    p2 = sort_rows(p2, s[0])
    wdist = math_ops.square(p1 - p2)
    return wdist

alpha_L1_norm_r = 0.0001
alpha_L1_norm_d = 0.00001
loss_r1_gt_norm = tf.reduce_mean(tf.square(r1_l - label)) + alpha_L1_norm_r * (tf.reduce_mean(tf.norm(rgb_W1 , ord=1)) + tf.reduce_mean(tf.norm(rgb_W2 , ord=1)))
loss_r2_gt_norm = tf.reduce_mean(tf.square(r2_l - label)) + alpha_L1_norm_d * (tf.reduce_mean(tf.norm(dep_W1 , ord=1)) + tf.reduce_mean(tf.norm(dep_W2 , ord=1)))
loss_r1_gt = tf.reduce_mean(tf.square(r1_l - label))
loss_r2_gt = tf.reduce_mean(tf.square(r2_l - label))
loss_r3_gt = tf.reduce_mean(tf.square(r3_l - label))

loss_r1_cv_ori_12 = tf.reduce_mean(discrepancy_slice_wasserstein(r1_u, r2_u), axis=1)
loss_r1_cv_ori_13 = tf.reduce_mean(discrepancy_slice_wasserstein(r1_u, r3_u), axis=1)
loss_r1_cv_ori_23 = tf.reduce_mean(discrepancy_slice_wasserstein(r2_u, r3_u), axis=1)
loss_r1_cv_12 = (1 + tf.reduce_mean(Entropy(r1_u)) / tf.reduce_mean(Entropy(r2_u))) * tf.reduce_mean(loss_r1_cv_ori_12)
loss_r2_cv_12 = (1 + tf.reduce_mean(Entropy(r2_u)) / tf.reduce_mean(Entropy(r1_u))) * tf.reduce_mean(loss_r1_cv_ori_12)

loss_r1_cv_13 = tf.reduce_mean(tf.multiply(tf.divide(Entropy(r1_u), Entropy(r3_u)), loss_r1_cv_ori_13)) + tf.reduce_mean(loss_r1_cv_ori_13)
loss_r3_cv_13 = tf.reduce_mean(tf.multiply(tf.divide(Entropy(r3_u), Entropy(r1_u)), loss_r1_cv_ori_13)) + tf.reduce_mean(loss_r1_cv_ori_13)

loss_r2_cv_23 = tf.reduce_mean(tf.multiply(tf.divide(Entropy(r2_u), Entropy(r3_u)), loss_r1_cv_ori_23)) + tf.reduce_mean(loss_r1_cv_ori_23)
loss_r3_cv_23 = tf.reduce_mean(tf.multiply(tf.divide(Entropy(r3_u), Entropy(r2_u)), loss_r1_cv_ori_23)) + tf.reduce_mean(loss_r1_cv_ori_23)

# # Input: predicted-RGB label and predicted-Depth label
def classifier(pr, pd, para):
    pr_in = pr 
    pd_in = pd

    classifier_W1 = para[0]
    classifier_b1 = para[1]
    classifier_W2 = para[2]
    classifier_b2 = para[3]

    C_prob_1 = tf.expand_dims(pr_in, -1)
    C_prob_1_ = tf.expand_dims(pr_in, 1)
    C_prob_2 = tf.expand_dims(pd_in, -1)
    C_prob_2_ = tf.expand_dims(pd_in, 1)

    W_feature_1 = tf.matmul(C_prob_1, C_prob_2_) 
    C_hw_1 = tf.reshape(W_feature_1, [-1, class_num * class_num])
    W_feature_2 = tf.matmul(C_prob_1, C_prob_1_) 
    C_hw_2 = tf.reshape(W_feature_2, [-1, class_num * class_num])
    W_feature_3 = tf.matmul(C_prob_2, C_prob_2_) 
    C_hw_3 = tf.reshape(W_feature_3, [-1, class_num * class_num])

    C_hw = C_hw_1 + C_hw_2 + C_hw_3
    C_h1 = leak_relu(tf.matmul(C_hw, classifier_W1) + classifier_b1, 0.25)
    Classifier_g_logit = tf.matmul(C_h1, classifier_W2) + classifier_b2 
    return Classifier_g_logit


num_filter = 3
size_kernel = [1,1]

classifier_W1 = tf.Variable(tf.truncated_normal([class_num*class_num, hidden_size], stddev=0.1), dtype=tf.float32)
classifier_b1 = tf.Variable(tf.constant(1.0, shape=[hidden_size]), dtype=tf.float32)
classifier_W2 = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
classifier_b2 = tf.Variable(tf.constant(1.0, shape=[class_num]), dtype=tf.float32)
para_cla = [classifier_W1, classifier_b1, classifier_W2, classifier_b2]

predict_l = classifier(r1_l, r2_l, para_cla)
predict_u = classifier(r1_u, r2_u, para_cla)
predict_u_1 = classifier(r1_u_1, r2_u_1, para_cla)
predict_u_2 = classifier(r1_u_2, r2_u_2, para_cla)

loss_repre_f = tf.reduce_mean(tf.square(label_loss - (predict_u_1 - predict_u_2)))
loss_predict = tf.reduce_mean(tf.square(predict_l - label))

loss_r1_gt_solver = tf.train.AdamOptimizer(learning_rate=lr_r).minimize(loss_r1_gt_norm, var_list=[rgb_W1, rgb_b1, rgb_W2, rgb_b2])
loss_r2_gt_solver = tf.train.AdamOptimizer(learning_rate=lr_d).minimize(loss_r2_gt_norm, var_list=[dep_W1, dep_b1, dep_W2, dep_b2])
loss_r3_gt_solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_r3_gt, var_list=[e1_w1_BN, e1_scale1, e1_beta1, e2_w1_BN, e2_scale1, e2_beta1, rd_W1, rd_b1, rd_W2, rd_b2])

loss_r1_cv_1_solver_12 = tf.train.AdamOptimizer(learning_rate=lr_r).minimize(loss_r1_cv_12 + loss_r1_gt + loss_predict, var_list=[rgb_W1, rgb_b1])
loss_r2_cv_1_solver_12 = tf.train.AdamOptimizer(learning_rate=lr_d).minimize(loss_r2_cv_12 + loss_r2_gt + loss_predict, var_list=[dep_W1, dep_b1])
loss_r1_cv_2_solver_12 = tf.train.AdamOptimizer(learning_rate=lr_r).minimize(-loss_r1_cv_12 + loss_r1_gt + loss_predict, var_list=[rgb_W2, rgb_b2])
loss_r2_cv_2_solver_12 = tf.train.AdamOptimizer(learning_rate=lr_d).minimize(-loss_r2_cv_12 + loss_r2_gt + loss_predict, var_list=[dep_W2, dep_b2])

loss_predict_solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_predict, var_list=[rgb_W1, rgb_b1, rgb_W2, rgb_b2, dep_W1, dep_b1, dep_W2, dep_b2, para_cla])

loss_repre_1_solver = tf.train.AdamOptimizer(learning_rate=lr_r).minimize(loss_repre_1 + loss_predict, var_list=[rgb_W1, rgb_b1, rgb_W2, rgb_b2])
loss_repre_2_solver = tf.train.AdamOptimizer(learning_rate=lr_d).minimize(loss_repre_2 + loss_predict, var_list=[dep_W1, dep_b1, dep_W2, dep_b2])
loss_repre_3_solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_repre_3, var_list=[rgb_W1, rgb_b1, dep_W1, dep_b1, e1_w1_BN, e1_scale1, e1_beta1, e2_w1_BN, e2_scale1, e2_beta1, rd_W1, rd_b1, rd_W2, rd_b2])

train_r = tf.argmax(r1_l, 1)
train_d = tf.argmax(r2_l, 1)
train_rd = tf.argmax(r3_l, 1)
test_r = tf.argmax(r1_u, 1)
test_d = tf.argmax(r2_u, 1)
test_rd = tf.argmax(r3_u, 1)
train_res = tf.argmax(predict_l, 1)
train_gt = tf.argmax(label, 1)
test_res = tf.argmax(predict_u, 1)
test_gt = tf.argmax(label_test, 1)

acc_train_r = tf.reduce_mean(tf.cast(tf.equal(train_r, train_gt), "float"))
acc_test_r = tf.reduce_mean(tf.cast(tf.equal(test_r, test_gt), "float"))
acc_train_d = tf.reduce_mean(tf.cast(tf.equal(train_d, train_gt), "float"))
acc_test_d = tf.reduce_mean(tf.cast(tf.equal(test_d, test_gt), "float"))
acc_train_rd = tf.reduce_mean(tf.cast(tf.equal(train_rd, train_gt), "float"))
acc_test_rd = tf.reduce_mean(tf.cast(tf.equal(test_rd, test_gt), "float"))
acc_train = tf.reduce_mean(tf.cast(tf.equal(train_res, train_gt), "float"))
acc_test = tf.reduce_mean(tf.cast(tf.equal(test_res, test_gt), "float"))

def main():
    
    # ============= Set the GPU usage ===========
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    print("begin")
    f_r_l_o, f_r_u_o, f_d_l_o, f_d_u_o, l_o, label2_o = train_data.DA_init()
    # train_data.find_ave_best()
    output_rgb = []
    output_dep = []
    output_g = []
    

    for i in range(1000):
        for train_i in range(50):
            f_r_l_1, f_d_l_1, l_1, _, _ = train_data.DA_train_next_batch()
            f_r_l_2, f_d_l_2, l_2, f_r_u, f_d_u = train_data.DA_train_next_batch()
            _alpha = np.random.beta(0.5, 0.5, 1)
            _alpha = np.minimum(_alpha, 1-_alpha)

            l_3 = np.array(l_1) * _alpha + np.array(l_2) * (1 - _alpha)
            _, _, _ = sess.run([loss_r1_gt_solver, loss_r2_gt_solver, loss_predict_solver], 
                feed_dict={rgb_features_l_1: f_r_l_1, depth_features_l_1: f_d_l_1, rgb_features_l_2: f_r_l_2, depth_features_l_2: f_d_l_2, label: l_3, alpha: _alpha, lr_r:0.0001, lr_d:0.00001, lr:0.00001})
            
            for _ in range(1):
                _label_loss = np.array(l_1) * (1 - _alpha) - np.array(l_2) * (1 - _alpha)
                _, _ = sess.run([loss_repre_1_solver, loss_repre_2_solver], feed_dict={
                    rgb_features_l_1: f_r_l_1, depth_features_l_1: f_d_l_1, rgb_features_l_2: f_r_l_2, depth_features_l_2: f_d_l_2, label: l_3, alpha: _alpha, 
                    rgb_features_u: f_r_u, depth_features_u: f_d_u, label_loss: _label_loss, lr_r:0.00001, lr_d:0.000001})
                
                # domain adapation
                _, _ = sess.run([loss_r1_cv_1_solver_12, loss_r2_cv_1_solver_12], feed_dict={
                    rgb_features_l_1: f_r_l_1, depth_features_l_1: f_d_l_1, rgb_features_l_2: f_r_l_2, depth_features_l_2: f_d_l_2, label: l_3, alpha: _alpha, 
                    rgb_features_u: f_r_u, depth_features_u: f_d_u, lr_r:0.00001, lr_d:0.00001})
                _, _ = sess.run([loss_r1_cv_2_solver_12, loss_r2_cv_2_solver_12], feed_dict={
                    rgb_features_l_1: f_r_l_1, depth_features_l_1: f_d_l_1, rgb_features_l_2: f_r_l_2, depth_features_l_2: f_d_l_2, label: l_3, alpha: _alpha, 
                    rgb_features_u: f_r_u, depth_features_u: f_d_u, lr_r:0.00001, lr_d:0.00001})


        # test
        acc_train_r_res, acc_train_d_res, acc_train_rd_res, acc_test_r_res, acc_test_d_res, acc_test_rd_res, acc_train_res, acc_test_res = sess.run([acc_train_r, acc_train_d, acc_train_rd, acc_test_r, acc_test_d, acc_test_rd, acc_train, acc_test], 
                feed_dict={rgb_features_l_1: f_r_l_o, depth_features_l_1: f_d_l_o, rgb_features_l_2: f_r_l_o, depth_features_l_2: f_d_l_o, 
                    rgb_features_u: f_r_u_o, depth_features_u: f_d_u_o, label: l_o, label_test:label2_o, alpha:1})
        output_rgb.append(acc_test_r_res)
        output_dep.append(acc_test_d_res)
        output_g.append(acc_test_res)
        print("iteration  %d" %i)
        print("rgb logits train: %.4f      test: %.4f" %(acc_train_r_res, acc_test_r_res))
        print("depth logits train: %.4f      test: %.4f" %(acc_train_d_res, acc_test_d_res))
        print("final logits train: %.4f      test: %.4f" %(acc_train_res, acc_test_res))

    # release GPU memory
    tf.reset_default_graph()

main()
