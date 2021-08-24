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
        t1 = time.perf_counter()
        value = func(*args, **kwargs)
        print(f"Time to run {func.__name__!r}: {time.perf_counter() - t1:.4f} secs")
        return value
    return wrapper 





@timeit
def apply_matrix(v):
  return jnp.dot(mat, v)

@timeit
def naively_batched_apply_matrix(v_batched):
  return jnp.stack([apply_matrix(v) for v in v_batched])

@timeit
@jit
def batched_apply_matrix(v_batched):
  return jnp.dot(v_batched, mat.T)

@timeit
@jit
def vmap_batched_apply_matrix(v_batched):
  return vmap(apply_matrix)(v_batched)




naively_batched_apply_matrix(batched_x).block_until_ready()
batched_apply_matrix(batched_x).block_until_ready()
vmap_batched_apply_matrix(batched_x).block_until_ready()






