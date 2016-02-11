import theano
import theano.tensor as T
import numpy as np
import sys
sys.path.insert(0, '../data_loader/')
import load

# load data
x_train, t_train, x_test, t_test = load.cifar10(dtype=theano.config.floatX)
labels_test = np.argmax(t_test, axis=1)

# define symbolic Theano variables
x = T.matrix()
t = T.matrix()

# define model: logistic regression
def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1))

def model(x, w):
    return T.nnet.softmax(T.dot(x, w))

w = init_weights((32 * 32, 10))

p_y_given_x = model(x, w)
y = T.argmax(p_y_given_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(p_y_given_x, t))
g = T.grad(cost, w)
updates = [(w, w - g * 0.001)]

# compile theano functions
train = theano.function([x, t], cost, updates=updates)
predict = theano.function([x], y)

# train model
batch_size = 50
for i in range(100):
    print "iteration {}".format(i + 1)
    for start in range(0, len(x_train), batch_size):
        x_batch = x_train[start:start + batch_size]
        t_batch = t_train[start:start + batch_size]
        cost = train(x_batch, t_batch)
    predictions_test = predict(x_test)
    accuracy = np.mean(predictions_test == labels_test)
    print "accuracy: {}".format(accuracy) + "\n"

