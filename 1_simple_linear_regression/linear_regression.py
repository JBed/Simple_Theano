import theano
import theano.tensor as T
import numpy as np

# create artificial training data
x_train = np.linspace(-1, 1, 101)
t_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.33


# define symbolic Theano variables
x = T.scalar()
t = T.scalar()

# define model: linear regression
# no bias term
def model(x, w):
    return x * w

w = theano.shared(0.0)
y = model(x, w)

cost = T.mean((t - y) ** 2)
g = T.grad(cost, w)
updates = [(w, w - g * 0.01)]


# compile theano function
train = theano.function([x, t], cost, updates=updates)


# train model
for ii in range(20):
    print "iteration {}".format(ii + 1)
    for x, t in zip(x_train, t_train):
        train(x, t)
    print "w = {}".format(w.get_value()) + '\n'

