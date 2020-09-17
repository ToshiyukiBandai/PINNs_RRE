#!/usr/bin/env python
# coding: utf-8

# uninstall tensorflow 2.x on Google Colab
# %tensorflow_version 2.x
# !pip uninstall -y tensorflow
# !pip install tensorflow-gpu==1.14.0

import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os
import random

tf.__version__ # tensorflow 1.4

# PhysicsInformedNN class
class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, t, z, theta, layers_psi, layers_theta, layers_K):
        """
        t: time training data
        z: space training data
        theta: water content training data
        layers_psi: nueral networks structure for capillary potential (ex. [2,20,20,1])
        layers_theta: nueral networks structure for water content (ex. [1,20,1])
        layers_K: nueral networks structure for hydraulic conductivity (ex. [1,20,1])
        """

        # Training data for system identification

        self.t= t
        self.z = z
        self.theta = theta

        # the structure of the three neural networks

        self.layers_psi = layers_psi
        self.layers_theta = layers_theta
        self.layers_K = layers_K

        # initialize NNs for the PINNs with monotonicity constraints
        self.weights_psi, self.biases_psi = self.initialize_NN(layers_psi)
        self.weights_theta, self.biases_theta = self.initialize_MNN(layers_theta)
        self.weights_K, self.biases_K = self.initialize_MNN(layers_K)

        # initialize NNs for the PINNs without monotonicity constraints
#         self.weights_psi, self.biases_psi = self.initialize_NN(layers_psi)
#         self.weights_theta, self.biases_theta = self.initialize_NN(layers_theta)
#         self.weights_K, self.biases_K = self.initialize_NN(layers_K)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        # tf placeholder

        self.z_tf = tf.placeholder(tf.float32, shape = [None, self.z.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape = [None, self.t.shape[1]])
        self.theta_tf = tf.placeholder(tf.float32, shape = [None, self.theta.shape[1]])
        self.psi_tf = tf.placeholder(tf.float32, shape = [None, self.theta.shape[1]])  # this is for lookup table

        # prediction from PINNs
        self.theta_pred, self.psi_pred, self.K_pred,         self.f_pred, self.theta_t_pred, self.psi_z_pred, self.psi_zz_pred, self.K_z_pred = self.net(self.t_tf, self.z_tf)

        # lookup table from PINNs
        tf.log_h = tf.math.log(-self.psi_tf)
        self.WRC_theta = self.net_theta(-tf.log_h, self.weights_theta, self.biases_theta)
        self.HCF_K = self.net_K(-tf.log_h, self.weights_K, self.biases_K)

        # loss for identification
        self.loss = tf.reduce_sum(tf.square(self.theta_tf - self.theta_pred)) + tf.reduce_sum(tf.square(self.f_pred))

        # Optimizer for identification
        # L-BFGS-B method
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                        method = 'L-BFGS-B',
                                                        options = {'maxiter': 50000,
                                                                   'maxfun': 50000,
                                                                   'maxcor': 50,
                                                                   'maxls': 50,
                                                                   'ftol' : 1.0 * np.finfo(float).eps})

        # Adam method
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        # tf.saver
        self.saver = tf.train.Saver()

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev = xavier_stddev), dtype = tf.float32)

    def initialize_NN(self, layers): # stndard neural network
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers-1):
            W = self.xavier_init(size = [layers[l],layers[l+1]])
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype = tf.float32), dtype = tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def initialize_MNN(self, layers): # monotonic neural network
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers-1):
            W = self.xavier_init(size = [layers[l],layers[l+1]])
            W2 = W**2
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype = tf.float32), dtype = tf.float32)
            weights.append(W2)
            biases.append(b)
        return weights, biases

    def net_psi(self, X, weights, biases): # NN for psi
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        psi = -tf.exp(Y)  # force psi to be negative
        return psi

    def net_theta(self, X, weights, biases):  # NN for theta
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        theta = tf.sigmoid(tf.add(tf.matmul(H, W), b)) # force theta to be between 0 and 1
        return theta

    def net_K(self, X, weights, biases):  # NN for K
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        K = tf.exp(tf.add(tf.matmul(H, W), b))  # force K to be positive
        return K

    def net(self, t, z):  # PINNs
        X = tf.concat([t, z],1)
        psi = self.net_psi(X, self.weights_psi, self.biases_psi)

        log_h = tf.math.log(-psi)
        theta = self.net_theta(-log_h, self.weights_theta, self.biases_theta)
        K = self.net_K(-log_h, self.weights_K, self.biases_K)

        theta_t = tf.gradients(theta, t)[0]
        psi_z = tf.gradients(psi, z)[0]
        psi_zz = tf.gradients(psi_z, z)[0]
        K_z = tf.gradients(K, z)[0]

        # residual for Richards equation
        f = theta_t - K_z*psi_z- K*psi_zz - K_z

        return theta, psi, K, f, theta_t, psi_z, psi_zz, K_z

    def train(self, N_iter):
        tf_dict = {self.t_tf: self.t, self.z_tf: self.z, self.theta_tf: self.theta}

        start_time = time.time()
        # Adam
        for it in range(N_iter):
            self.sess.run(self.train_op_Adam, tf_dict)

            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %(it, loss_value, elapsed))
                start_time = time.time()

        # L-BFGS-B
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)

        loss_value = self.sess.run(self.loss, tf_dict)

    def callback(self, loss):
        print('Loss: %.3e' %(loss))

    def WRC_HCF(self, psi_star):
        tf_dict = {self.psi_tf: psi_star}

        theta = self.sess.run(self.WRC_theta, tf_dict)
        K = self.sess.run(self.HCF_K, tf_dict)
        return theta, K

    def save_model(self, path):

        save_path = self.saver.restore(self.sess, path)
        print("Model saved in path: %s" % save_path)


    # wight and bias parameters of the NNs
    def PINNs_parameters(self):

        weights_psi_star = self.sess.run(self.weights_psi)
        biases_psi_star = self.sess.run(self.biases_psi)

        weights_theta_star = self.sess.run(self.weights_theta)
        biases_theta_star = self.sess.run(self.biases_theta)

        weights_K_star = self.sess.run(self.weights_K)
        biases_K_star = self.sess.run(self.biases_K)

        return weights_psi_star, biases_psi_star, weights_theta_star, biases_theta_star, weights_K_star, biases_K_star

    def predict(self, t_star, z_star):
        tf_dict = {self.t_tf: t_star, self.z_tf: z_star}

        theta_star = self.sess.run(self.theta_pred, tf_dict)
        psi_star = self.sess.run(self.psi_pred, tf_dict)
        K_star = self.sess.run(self.K_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)
        theta_t_star = self.sess.run(self.theta_t_pred, tf_dict)
        psi_z_star = self.sess.run(self.psi_z_pred, tf_dict)
        psi_zz_star = self.sess.run(self.psi_zz_pred, tf_dict)
        K_z_star = self.sess.run(self.K_z_pred, tf_dict)
        flux_star = -K_star*(psi_z_star + 1.0)

        return theta_star, psi_star, K_star, f_star, theta_t_star, psi_z_star, psi_zz_star, K_z_star, flux_star

def main_loop(hydrus, depth_increment, noise, num_layers_psi, num_neurons_psi, num_layers_theta, num_neurons_theta, num_layers_K, num_neurons_K, number_random):
    """
    hydrus: HYDRUS data type ("sandy_loam", "loam", "silt_loam", "sandy_loam2", "loam2", "silt_loam2")
    noise: the standard deviation of noise added to the synthetic data
    depth_increment: 1 or 2 or 3. 1 means every 2 cm, 2, means every 4 cm, 3 ,eams: 6cm, 4 means: 8 cm increment
    num_layers_psi: the number of hidden layers for psi NN
    num_neurons_psi: the number of units for psi NN
    num_layers_theta: the number of hidden layers for theta NN
    num_neurons_theta: the number of units for theta NN
    num_layers_K: the number of hidden layers for K NN
    num_neurons_K: the number of units for K NN
    number_random: random seeds
    """

    # reset the graph and set random seeds
    tf.reset_default_graph()
    tf.set_random_seed(0)
    random.seed(0)
    np.random.seed(0)

    # import HYDRUS_nod data
    data = pd.read_csv(f"./Node_Inf/{hydrus}_nod.csv")
    t = data['time'].values[:,None]
    z = data['depth'].values[:,None]
    psi = data['head'].values[:,None]
    K = data['K'].values[:,None]
    C = data['C'].values[:,None]
    theta = data['theta'].values[:,None]
    flux = data['flux'].values[:,None]

    # raw data
    Z_star = np.hstack((t, z))
    theta_star = theta.flatten()[:,None]
    psi_star = psi.flatten()[:,None]
    K_star = K.flatten()[:,None]
    C_star = C.flatten()[:,None]
    flux_star = flux.flatten()[:,None]

    t_star = Z_star[:,0:1]
    z_star = Z_star[:,1:2]

    # interpolate predicted values (actually, it does not interpolate. The data point are the same as the coordinate.)
    space_nodes = 1001
    time_nodes = 251
    Z = z_star.reshape(time_nodes, space_nodes)
    T = t_star.reshape(time_nodes, space_nodes)

    # making lists for NN architectures
    layers_psi = np.concatenate([[2], num_neurons_psi*np.ones(num_layers_psi), [1]]).astype(int).tolist()
    layers_theta = np.concatenate([[1], num_neurons_theta*np.ones(num_layers_theta), [1]]).astype(int).tolist()
    layers_K = np.concatenate([[1], num_neurons_K*np.ones(num_layers_K), [1]]).astype(int).tolist()

    fixed_position_full = [-0.05, -0.15, -0.25, -0.35, -0.45, -0.55, -0.65, -0.75, -0.85, -0.95]  # dimensionless depth
    fixed_position = fixed_position_full[::depth_increment] # change the number of virtual sensors
    for i in range(len(fixed_position)):
        if i == 0:
            fixed_list = data.index[data['zeta'] == fixed_position[i]].values
        else:
            fixed_list = np.append(fixed_list, data.index[data['zeta'] == fixed_position[i]].values)

    # adding noise to theta

    noise_theta = noise*np.random.randn(theta_star.shape[0], theta_star.shape[1]) # standard normal distibuiton with 0 mean and stndard error of the noise value
    theta_noise = theta_star + noise_theta

    # fixed points (dimensionless and raw)
    Z_train = Z_star[fixed_list,:]
    theta_train = theta_noise[fixed_list, :]
    psi_train = psi_star[fixed_list, :]
    K_train = K_star[fixed_list, :]

    # training data
    t_train = Z_train[:, 0:1]
    z_train = Z_train[:, 1:2]

    run = hydrus + "_depth_" + str(depth_increment) + "_noise_" + str(noise) + "_lay_psi_" + str(num_layers_psi) + "_neu_psi_" + str(num_neurons_psi) + "_lay_theta_" + str(num_layers_theta) + "_neu_theta_" + str(num_neurons_theta) + "_lay_K_" + str(num_layers_K) + "_neu_K_" + str(num_neurons_K) + "_random_" + str(number_random)
    path = f'{run}'
    train_data = pd.DataFrame({'z': z_train.flatten(), 't': t_train.flatten(),
                        'theta_train': theta_train.flatten()})

    train_data.to_csv(f"./results/{hydrus}/{run}/train_data.csv")

    # random seeds
    tf.reset_default_graph()
    tf.set_random_seed(number_random)
    random.seed(number_random)
    np.random.seed(number_random)

    model = PhysicsInformedNN(t_train, z_train, theta_train, layers_psi,layers_theta, layers_K)

    model.train(1000)
    print(f'run is {run}')

    theta_pred, psi_pred, K_pred, f_pred, theta_t_pred, psi_z_pred, psi_zz_pred, K_z_pred, flux_pred = model.predict(t_star, z_star)

    # dataset
    dataset = pd.DataFrame({'z': z_star.flatten(), 't': t_star.flatten(),
                        'theta_actual': theta_star.flatten(), 'theta_pred': theta_pred.flatten(),
                        'theta_noise': theta_noise.flatten(),
                        'psi_actual': psi_star.flatten(), 'psi_pred': psi_pred.flatten(),
                        'K_actual': K_star.flatten(), 'K_pred': K_pred.flatten(),
                        'flux_actual': flux_star.flatten(), 'flux_pred': flux_pred.flatten(),
                        'f_pred': f_pred.flatten(), 'theta_t_pred': theta_t_pred.flatten(),
                        'psi_z_pred': psi_z_pred.flatten(), 'psi_zz_pred': psi_zz_pred.flatten(),
                        'K_z_pred': K_z_pred.flatten()})

    dataset.to_csv(f"./results/{hydrus}/{run}/data.csv")

    # store parameters from the trained PINNs
    weights_psi_star, biases_psi_star, weights_theta_star, biases_theta_star, weights_K_star, biases_K_star = model.PINNs_parameters()
    print("weights_psi", weights_psi_star)
    print("biases_psi", biases_psi_star)
    print("weights_theta", weights_theta_star)
    print("biases_theta", biases_theta_star)
    print("weights_K", weights_K_star)
    print("biases_K", biases_K_star)

    ## lookup table
    log_h_look = np.arange(-5, 3.4, 0.021)
    h_look = 10**log_h_look
    psi_look = -h_look.reshape(400,1)
    theta_look, K_look = model.WRC_HCF(psi_look)

    lookup = pd.DataFrame({'theta': theta_look.flatten(), 'psi': psi_look.flatten(),
                           'K': K_look.flatten()})

    lookup.to_csv(f"./results/{hydrus}/{run}/lookup.csv", index = False)

# for google Colab
# from google.colab import drive
# drive.mount("/content/drive")


hydrus = 'sandy_loam'
noise = [0]
depth_increment = [1, 2, 3] # depths increment: 1 means every 2 cm, 2, means every 4 cm, 3 ,eams: 6cm, 4 means: 8 cm
num_layers_psi = [8]
num_neurons_psi = [40]
num_layers_theta = [1, 2, 3]
num_neurons_theta = [10, 20, 40]
num_layers_K = [1, 2, 3]
num_neurons_K = [10, 20, 40]
number_random = [111]

main_loop(hydrus, depth_increment[0], noise[0], num_layers_psi[0], num_neurons_psi[0], num_layers_theta[0], num_neurons_theta[0], num_layers_K[i], num_neurons_K[j], number_random[0])
