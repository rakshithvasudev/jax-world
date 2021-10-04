import numpy as onp
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax import random

key = random.PRNGKey(1)


batch_dim = 32
feature_dim = 100
hidden_dim = 512

X = random.normal(key, (batch_dim, feature_dim))
print(X.shape)

params = [random.normal(key, (hidden_dim, feature_dim)),
        random.normal(key, (hidden_dim, ))]

def ReLU(x):
    return np.maximum(0, x)

def relu_layer(params, x):
    return ReLU(np.dot(params[0], x) + params[1])


def batch_version_relu_layer(params, x):
    return ReLU(np.dot(x, params[0].T) + params[1])

def vmap_relu_layer(params, x):
    return jit(vmap(relu_layer, in_axes=(None, 0), out_axes=0))



