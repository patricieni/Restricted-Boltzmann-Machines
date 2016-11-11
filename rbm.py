import csv
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
from dateutil.parser import parse
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema

import numpy as np
from numpy import random as rng
from sklearn import linear_model
from sklearn import pipeline
import theano

import pandas as pd
import os


# Implement Restricted Boltzmann Machines using numpy
# This is level 1, level 2 will be Tensor Flow.

class RBM:
    '''

    Iniatilize the RBM with a number of visible and hidden units and the usual bias terms, considered zero to start off
            n_visible = number of visible units
            n_hidden  = number of hidden units
            v_bias = visible bias units
            h_bias = hidden bias units
            l_rate = learning rate of Contrastive Divergence algorithm
            weights = weights between the visible and hidden units
    '''

    def __init__(self, n_visible, n_hidden, weights, l_rate=0.1):
        self.n_hidden = n_hidden
        self.n_visible = n_visible
        self.l_rate = l_rate

        # Initialize some random weights to all connections, assume a bipartite graph for now.
        # Use heuristics from Hinton's Guide for weights and set biases to zero
        self.weights = weights
        self.h_bias = np.zeros(n_hidden)
        self.v_bias = np.zeros(n_visible)

    # Alternate to see whether you get the desired input through the RBM.
    def daydream(self, benchmark, d):
        i = 0
        while i < benchmark:
            [h_units, h_p] = self.compute_hidden_units(d)
            [v_units, v_p] = self.reconstruct_visible_units(h_units)
            d = v_units
            i += 1
        return v_units

    # Perform Gibbs Sampling from the joint distribution(the one we learnt)
    # After a few steps we get the reconstructed data from the partial data.
    # This method uses the binary data to find hidden units
    # This is the best one at reconstructing!
    def infer_data(self, steps, partial_data):
        v_units = np.ones(self.n_visible)
        p_visible = np.ones(self.n_visible)
        # Assume you have incomplete data. Find the hidden units for the incomplete data. for each connection..
        for step in range(0, steps):
            [h_units, h_p] = self.compute_hidden_units(partial_data)
            for i in range(0, self.n_visible):
                d = partial_data[i]
                p_visible[i] = d
                v_units[i] = d
                if (d != 1) and (d != 0):
                    dot_product = np.dot(h_units, self.weights[i, :])
                    p_visible[i] = sigmoid(self.v_bias[i] + dot_product)
                    v_units[i] = np.array(p_visible[i] >= rng.rand()).astype(int)
            partial_data = v_units
        return v_units

    # This method uses probabilities to compute hidden units.
    def infer_data_1(self, steps, partial_data):
        v_units = np.ones(self.n_visible)
        p_visible = np.ones(self.n_visible)
        # Assume you have incomplete data. Find the hidden units for the incomplete data. for each connection..
        for step in range(0, steps):
            [h_units, h_p] = self.compute_hidden_units(partial_data)
            for i in range(0, self.n_visible):
                d = partial_data[i]
                p_visible[i] = d
                v_units[i] = d
                if (d != 1) and (d != 0):
                    dot_product = np.dot(h_units, self.weights[i, :])
                    p_visible[i] = sigmoid(self.v_bias[i] + dot_product)
                    a = np.array(p_visible[i] >= rng.rand()).astype(int)
                    v_units[i] = a
            partial_data = p_visible
        return v_units

    # This one does it in reverse. First decides if it's a zero or one, then computes the hidden units via binary values
    # This one doesn't really do the job
    def infer_data_2(self, steps, partial_prob):
        p_visible = np.ones(self.n_visible)
        for step in range(0, steps):
            # Sample the unknown variables
            v_units = np.array(partial_prob >= rng.rand()).astype(int)
            # Find h based on the samped values
            [h_units, h_p] = self.compute_hidden_units(v_units)
            # Calculate new probabilities to sample using the joint distribution
            for i in range(0, self.n_visible):
                d = partial_prob[i]
                p_visible[i] = d
                if (d != 1) and (d != 0):
                    dot_product = np.dot(h_units, self.weights[i, :])
                    p_visible[i] = sigmoid(self.v_bias[i] + dot_product)
            partial_prob = p_visible
        v_units = np.array(partial_prob >= rng.rand()).astype(int)
        return v_units

    def run_contrastive_divergence(self, K, data, epochs):
        """
        :return: Update the weights using contrastive divergence (and the two biases as well)
        """
        size = data.shape[0]
        error = np.zeros(epochs)
        # Iterate over all the training data and then perform k step on each data input
        for epoch in range(0, epochs):
            # Shuffle data to get unbiased samples on each epoch/iteration
            np.random.shuffle(data)
            for i in range(0, size):
                initial_data = data[i]
                initial_h_p = np.zeros(self.n_hidden)
                final_v_p = np.zeros(self.n_visible)
                for k in range(0, K):
                    [h_units, h_p] = self.compute_hidden_units(initial_data)
                    [v_units, v_p] = self.reconstruct_visible_units(h_units)
                    if k == 0:
                        initial_h_p = h_p
                    initial_data = v_units
                    final_v_p = v_p

                # Multiply hidden probabilities with the input data vector
                positive_divergence = rbm_multiplier(data[i], initial_h_p)

                # Compute the last hidden unit to get your probabilities
                [final_h_unit, final_h_p] = self.compute_hidden_units(initial_data)

                # Compute negative divergence using kth reconstructed probabilities and hidden probabilities
                negative_divergence = rbm_multiplier(final_v_p, final_h_p.T)

                # CD_k
                self.weights += self.l_rate * ((positive_divergence - negative_divergence) / size)
                self.h_bias += (initial_h_p - final_h_p)
                self.v_bias += (data[i] - initial_data)

                # MSE for input data and reconstructed data, this is plotted later
                error[epoch] += np.sum((data[i] - final_v_p) ** 2)
            epoch += 1
        return error

    def compute_hidden_units(self, training_data):

        """
        :param self: RBM containing number of hidden and visible units, self.n_visible needs to be the same as size(training_data)
        :param training_data: A vector containing visible units that are either 0/1
        :return: An activated/non-activated unit
        """
        # Create the vector that contains hidden units. Compute dot product for training data with the weights.
        # TODO: refactor loops into matrix multiplication by appending biases to the weight matrix
        h_units = np.ones(self.n_hidden)
        p_hidden = np.ones(self.n_hidden)

        for i in range(0, self.n_hidden):
            dot_product = np.dot(training_data, self.weights[:, i])
            p_hidden[i] = sigmoid(self.h_bias[i] + dot_product)

            # Sample to activate unit
            a = np.array(p_hidden[i] >= rng.rand()).astype(int)
            h_units[i] = np.multiply(h_units[i], a)
        return [h_units, p_hidden]

    def reconstruct_visible_units(self, hidden_units):
        """
        :param self: RBM
        :param hidden_units: Hidden units used to reconstruct the original ones.
        :return: a set of activated/non-activated visible units.
        """
        v_units = np.ones(self.n_visible)
        p_visible = np.ones(self.n_visible)

        for i in range(0, self.n_visible):
            dot_product = np.dot(hidden_units, self.weights[i, :])
            p_visible[i] = sigmoid(self.v_bias[i] + dot_product)
            a = np.array(p_visible[i] >= rng.rand()).astype(int)
            v_units[i] = np.multiply(v_units[i], a)
        return [v_units, p_visible]


# My own multiplier for counting over all the connections in the network.
# TODO: once we go into Matrix world we don't need this, it resumes to just simple matrix multiplication
def rbm_multiplier(np1, np2):
    val = np.ones((np1.size, np2.size))
    for i in range(0, np1.size):
        val[i] = np.multiply(np1[i], np2)
    return val


# Not used at this stage, it's the Hinton way of initializing the biases
def init_visible_bias(data):
    vis_bias = np.zeros(data.T.shape[0])
    for i in range(0, data.T.shape[0]):
        d = data[:, i]
        p = sum(d) / d.size
        q = np.log(p / (1 - p))
        if (q != np.inf) | (q != -np.inf):
            vis_bias[i] = q
    return vis_bias


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def error_plot(val, epochs):
    b = np.arange(0, epochs)
    plt.interactive(False)
    plt.plot(b, val, zorder=2)
    plt.draw()


# Generates the set of all Bars-As-Stripes matrices and reshapes them into a vector
def generate_artificial_bas(rng):
    all_data = np.empty(shape=16)
    size = 4
    big_enough = 0
    while big_enough < 500:
        data_i = np.zeros(shape=(4, 4))
        if rng.uniform() < 0.5:
            # to see whether we fill horizontally
            # direction = horizontal
            for s in range(0, size):
                if rng.uniform() < 0.5:
                    data_i[s] = np.zeros(shape=size)
                else:
                    data_i[s] = np.ones(shape=size)
            all_data = np.vstack([all_data, data_i.reshape(-1)])
        else:
            # direction = vertical
            for s in range(0, size):
                if rng.uniform() < 0.5:
                    data_i[:, s] = np.zeros(shape=size)
                else:
                    data_i[:, s] = np.ones(shape=size)
            all_data = np.vstack([all_data, data_i.reshape(-1)])
        big_enough += 1
    # uniqueness
    y = np.vstack({tuple(row) for row in all_data})
    return y


if __name__ == '__main__':
    print("Generating RBM with random weights and zero biases...")
    n_visible = 16
    n_hidden = 16
    numpy_rng = rng.RandomState(1234)
    weights = numpy_rng.uniform(size=(n_visible, n_hidden),
                                low=-4 * np.sqrt(6. / (n_visible + n_hidden)),
                                high=4 * np.sqrt(6. / (n_visible + n_hidden)))

    data = generate_artificial_bas(numpy_rng)

    rbm = RBM(n_visible=n_visible, n_hidden=n_hidden, weights=weights, l_rate=0.1)

    # Train the RBM to learn the weights and biases
    print('Training RBM using Contrastive Divergence: ', 1)
    err = rbm.run_contrastive_divergence(K=1, data=data, epochs=4000)
    error_plot(err, 4000)

    # input_data = np.array([[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0],
    #                [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0]])
    input_data = np.array([1, 0.5, 0.5, 0.5,
                           0, 0.5, 0.5, 0.5,
                           1, 0.5, 0.5, 0.5,
                           0, 0.5, 0.5, 0.5])
    input_data_1 = np.array([0, 0.5, 0.5, 0.5,
                             1, 0.5, 0.5, 0.5,
                             0, 0.5, 0.5, 0.5,
                             1, 0.5, 0.5, 0.5])
    input_data_2 = np.array([1, 0, 0, 1,
                             0.5, 0.5, 0.5, 0.5,
                             0.5, 0.5, 0.5, 0.5,
                             0.5, 0.5, 0.5, 0.5
                             ])

    input_data_test = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    input_data_1_test = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    input_data_2_test = np.array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1])

    # Just to see what the hidden units look like after learning
    # [values, probabilities] = rbm.compute_hidden_units(input_data_2)
    # print("Hidden unit values for some input after learning weights: ", values)


    # Learn the data by providing partial input - using infer_data_1 algorithm
    errors = np.zeros(15)
    for i in range(0, 15):
        reconstructed_data = rbm.infer_data(i, input_data)
        errors[i] = np.sum((input_data_test - reconstructed_data) ** 2)
    print("Reconstructed Data: ", reconstructed_data)
    print("Distance from truth", errors)

    errors_1 = np.zeros(15)
    for i in range(0, 15):
        reconstructed_data = rbm.infer_data(i, input_data_1)
        errors_1[i] = np.sum((input_data_1_test - reconstructed_data) ** 2)
    print("Reconstructed Data: ", reconstructed_data)
    print("Distance from truth", errors_1)

    # use the last algorithm
    errors_2 = np.zeros(15)
    for i in range(0, 15):
        reconstructed_data = rbm.infer_data(i, input_data_2)
        errors_2[i] = np.sum((input_data_2_test - reconstructed_data) ** 2)
    print("Reconstructed Data: ", reconstructed_data)
    print("Distance from truth", errors_2)

    # In order to detect the "image" we would need to train a logistic classifier for example on the hidden units
    # lgReg = linear_model.LogisticRegression(C=100)

    # See what all of the input gives you after 1 reconstruction. Should get good results?? As good as the error...
    '''print("After daydreaming for 15 cycles...")
    reconstruction = rbm.daydream(3000, input_data)
    print("Reconstructed Data: ", reconstruction)'''

    plt.show()
