import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import time
import functools

key = random.PRNGKey(0)

mat = random.normal(key, (150, 100))
batched_x = random.normal(key, (10, 100))

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        value = func(*args, **kwargs)
        print(f"Time to run {func.__name__!r}: {time.time() - t1:.4f} secs")
        return value
    return wrapper 





@timeit
def apply_matrix(v):
  return jnp.dot(mat, v)

@timeit
def naively_batched_apply_matrix(v_batched):
  return jnp.stack([apply_matrix(v) for v in v_batched])

#print('Naively batched')
#print(naively_batched_apply_matrix(batched_x).block_until_ready())
naively_batched_apply_matrix(batched_x).block_until_ready()






