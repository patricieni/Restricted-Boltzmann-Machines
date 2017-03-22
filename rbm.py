import time
import numpy as np
from numpy import random as rng

import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.externals.six.moves import xrange
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import gen_even_slices
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# Implementation taken from sk-learn to understand better PCD..."Persistent Contrastive Divergence"
# Took out the free energy and pseudo-likelihood calculation from the fitting algo = PCD
# Parametrize gibbs method, perform n sampling steps.
class RBM(BaseEstimator, TransformerMixin):
    '''

    Iniatilize the RBM with a number of visible and hidden units and the usual bias terms, considered zero to start off
            n_visible = number of visible units
            n_hidden  = number of hidden units
            v_bias = visible bias units
            h_bias = hidden bias units
            l_rate = learning rate of Contrastive Divergence algorithm
            weights = weights between the visible and hidden units
            Attributes: Weights and Biases
    '''

    def __init__(self,  n_hidden=256, learning_rate=0.1, batch_size=10,
                 n_iter=10, verbose=0, random_state=None):
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state

    def transform(self, X):
        """
        Returns the representation of the whole input data
        :param X: All the data to be transformed shape = (samples, n_hidden)
        :return: h
        """
        check_is_fitted(self, "weights_")
        check_array(X, accept_sparse='csr', dtype=np.float64)
        return self._mean_hiddens(X)

    def _mean_hiddens(self, v):

        p = safe_sparse_dot(v, self.weights_.T)
        p += self.bias_hidden_
        return sigmoid(p)

    def _sample_hidden_units(self, v):
        """
        :param v: A vector containing visible units that are either 0/1
        :return: The hidden units and their associated probabilities
        """

        p = self._mean_hiddens(v)
        h_units = rng.random_sample(size=p.shape) < p
        return h_units

    def _sample_visible_units(self, h):
        """
        :param h: A vector containing hidden units that are either 0/1
        :return: The hidden units and their associated probabilities
        """
        p_visible = np.dot(h, self.weights_)
        p_visible += self.bias_visible_
        p = sigmoid(p_visible)
        v_units = rng.random_sample(size=p.shape) < p
        return v_units


    def partial_fit(self, X, y = None):
        """
        Fit the RBM to the partial data X.
        Instantiate the weights and random state if they don't exist at this stage
        :param X: partial data to train
        :param y:
        :return: the RBM
        """
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        if not hasattr(self, 'random_state_'):
            self.random_state_ = check_random_state(self.random_state)
        if not hasattr(self, 'weights_'):
            self.weights_ = np.asarray(
                self.random_state_.normal(
                    0,
                    0.01,
                    (self.n_hidden, X.shape[1])
                ),
                order='F')
        if not hasattr(self, 'bias_hidden_'):
            self.bias_hidden_ = np.zeros(self.n_hidden, )
        if not hasattr(self, 'intercept_visible_'):
            self.bias_visible_ = np.zeros(X.shape[1], )
        if not hasattr(self, 'h_samples_'):
            self.h_samples_ = np.zeros((self.batch_size, self.n_hidden))

        self._fit(X)

    def _fit(self, v_pos, y=None):
        """Inner fit for one mini-batch.
        Adjust the parameters to maximize the likelihood of v using
        Stochastic Maximum Likelihood (SML).
        Parameters
        ----------
        v_pos : array-like, shape (n_samples, n_features)
            The data to use for training.
        rng : RandomState
            Random number generator to use for sampling.
        """
        h_prob = self._mean_hiddens(v_pos)
        v_neg = self._sample_visible_units(self.h_samples_)
        h_neg = self._mean_hiddens(v_neg)

        lr = float(self.learning_rate) / v_pos.shape[0]
        positive_div = safe_sparse_dot(v_pos.T, h_prob, dense_output=True).T
        negative_div = np.dot(h_neg.T, v_neg)
        update = positive_div - negative_div
        self.weights_ += lr * update

        self.bias_hidden_ += lr * (h_prob.sum(axis=0) - h_neg.sum(axis=0))
        self.bias_visible_ += lr * (np.asarray(
            v_pos.sum(axis=0)).squeeze() -
                                         v_neg.sum(axis=0))

        h_neg[rng.uniform(size=h_neg.shape) < h_neg] = 1.0  # sample binomial
        self.h_samples_ = np.floor(h_neg, h_neg)

    def fit(self, X, y=None):
        """Fit the model to the data X.
        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            Training data.
        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        n_samples = X.shape[0]
        rng = check_random_state(self.random_state)

        #self.weights_ = rng.uniform(size=(self.n_hidden, X.shape[1]),
        #                           low=-4 * np.sqrt(6. / (X.shape[1] + self.n_hidden)),
        #                          high=4 * np.sqrt(6. / (X.shape[1] + self.n_hidden)))
        self.weights_ = np.asarray(
            rng.normal(0, 0.01, (self.n_hidden, X.shape[1])),
            order='F')

        self.bias_hidden_ = np.zeros(self.n_hidden, )
        self.bias_visible_ = np.zeros(X.shape[1], )
        self.h_samples_ = np.zeros((self.batch_size, self.n_hidden))

        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(gen_even_slices(n_batches * self.batch_size,
                                            n_batches, n_samples))
        begin = time.time()
        for iteration in xrange(1, self.n_iter + 1):
            for batch_slice in batch_slices:
                self._fit(X[batch_slice])
        end = time.time()

        return self

    def gibbs(self, n_steps, v):
        '''
        Perform n Gibbs sampling steps
        After a few iterations we should get the reconstructed data back if the model is trained accordingly.
        :param n_steps: n steps of Gibbs Sampling
        :param v: visible input units
        :return: sampled visible units
        '''
        check_is_fitted(self, "weights_")
        if not hasattr(self, "random_state_"):
            self.random_state_ = check_random_state(self.random_state)
        for step in range(0, n_steps):
            h_ = self._sample_hidden_units(v)
            v_ = self._sample_visible_units(h_)
            v = v_
        return v


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
    all_data = np.zeros(shape=16)
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
    numpy_rng = rng.RandomState(123456)
    #data = generate_artificial_bas(numpy_rng)
    digits = datasets.load_digits()
    X = np.asarray(digits.data, 'float32')
    Y = digits.target
    X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

    # Split the data into training set and test set 0.8/0.2
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=0)
    rbm = RBM(random_state=numpy_rng)
    logistic = linear_model.LogisticRegression()

    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    rbm.n_hidden = 100
    logistic.C = 6000

    pipeline = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])


    pipeline.fit(X_train, Y_train)
    print()
    print("Pipeline:Logistic regression using RBM features:\n%s\n" % (
        metrics.classification_report(
            Y_test,
            pipeline.predict(X_test))))

