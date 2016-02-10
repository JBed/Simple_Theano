import theano
import theano.tensor as T

a = T.scalar()
b = T.scalar()

#theano.function(params, quation)
f = theano.function([a, b], a * b)

print f(1, 2)
print f(3, 3)

