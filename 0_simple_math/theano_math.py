import theano
import theano.tensor as T

a = T.scalar()
b = T.scalar()

#theano.function(params, quation)
f = theano.function([a, b], a * b)

print f(1, 2)
print f(3, 3)


# or 

a = T.scalar()

def power(base):
    return base**2

po = power(a)
f = theano.function([a], po)

print f(3)


# or
v = T.dvector(name='vector_v')
A = T.dmatrix(name='matrix_A')
print A
print A.type

#We can of do linear algebra.
Av = T.dot(A,v)
f = theano.function([A,v], [Av])


#or 

w=T.dvector(name='w_vector')
vTw = T.dot(v,w)
 
#Now we take gradients.
vTw_grad = T.grad(vTw,w)

import numpy as np
vec1 = np.asarray([1,2])
vec2 = np.asarray([0,0])
 
print vTw_grad.eval({w:vec1,v:vec2})

