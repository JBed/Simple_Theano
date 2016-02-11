
# take a look at some of the CIFAR-10 images
import load
import matplotlib.pyplot as plt
plt.ion()

x_train, t_train, x_test, t_test = load.cifar10(dtype=theano.config.floatX)

for ii in range(len(x_train)): 
	plt.imshow(x_train[0].reshape(32, 32), cmap=plt.cm.gray)
