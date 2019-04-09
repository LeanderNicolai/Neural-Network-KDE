from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
import numpy as np
from math import e
from math import log as log


def sigmoid(x):
    return 1 / (1 + e**(-x))


def loss(ytrue, ypred):
    return -((ytrue*np.log(ypred)) + ((1 - ytrue)*np.log(1 - ypred)))


def feed_forward(X, outer_weights, inner_weights):
    inner_prod = np.dot(X, inner_weights)
    hidden_out = sigmoid(inner_prod)
    hidden_out_wb = np.hstack([hidden_out, np.ones((hidden_out.shape[0], 1))])
    to_y = np.dot(hidden_out_wb, outer_weights)
    yh_sig = sigmoid(to_y)
    return hidden_out, hidden_out_wb, yh_sig


def der_sig(y):
    return sigmoid(y) * (1-sigmoid(y))


def back_prop(X, hidden_out, ypred, ytrue, outer_weights):
    y_loss = loss(ytrue, ypred)
    error = (ypred - ytrue) * loss(ytrue, ypred)
    grad_y = der_sig(ypred) * error
    LR = 0.18
    #hidden_out = np.hstack([hidden_out, np.ones((hidden_out.shape[0], 1))])
    weights_delta = np.dot(-grad_y, hidden_out) * LR
    print(weights_delta.shape)
    return grad_y, weights_delta


Xc, ytrue = make_moons(n_samples=50, noise=0.2, random_state=42)
X = np.hstack([Xc, np.ones((Xc.shape[0], 1))])
sig_x = sigmoid(X)

loss_plot = []
x = []
wh1_li = []
wh2_li = []
wo1_li = []
wo2_li = []
losses = []
for i in range(30):
    if i == 0:
        outer_weights = np.random.random(size=(3, 1))
        inner_weights = np.random.random(size=(3, 2))
        dec_loss = 30
    hidden_out, hidden_out_wb, ypred = feed_forward(X, outer_weights, inner_weights)
    ypred = ypred.reshape(50)
    grad_y, delta_w = back_prop(X, hidden_out_wb, ypred, ytrue, outer_weights)
    outer_weights = outer_weights.reshape(3)
    outer_weights = outer_weights + delta_w
    outer_weights_rs = outer_weights.reshape(1, 3)
    grad_y_rs = grad_y.reshape(50, 1)
    right_h = np.dot(grad_y_rs, outer_weights_rs)
    grad_ho = hidden_out_wb * right_h
    grad_ho = grad_ho.reshape(3, 50)
    LR = 0.05
    delta_wi = np.dot(-grad_ho, Xc) * LR
    inner_weights = inner_weights + delta_wi
    desc_loss = loss(ytrue, ypred).sum()
    loss_plot.append(desc_loss)
    x.append(i)
    print('The new loss is: ', desc_loss)
    print('new inner weights are: ', inner_weights)
    print('new outer weights are: ', outer_weights)
    print('iteration is: ', i)
    print('\n\n')
    losses.append(desc_loss)
    ypred[ypred >= 0.5] = 1
    ypred[ypred <= 0.5] = 0
    ypred = ypred.astype(int)
    if i > 10:
        plt.scatter(X[:, 0], X[:, 1], c=ypred)
        plt.show()
        if losses[-10] < losses[i]:
            break

plt.plot(x, loss_plot)
plt.show()
