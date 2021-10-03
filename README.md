# Jax Land
Welcome! Lets explore the beautiful world of Jax environment together :)



## Dockerfile Usage
The dockerfile has your fav libraries. JAX, Huggingface, PyTorch, Flax, Haiku and much more.
Right now this also includes torch for comparision and exploration. Functorch will be added soon.
Install [Docker](https://docs.docker.com/get-docker/) on your local host machine.

- For GPU support on Linux, [install NVIDIA Docker support](https://github.com/NVIDIA/nvidia-docker).
   - Take note of your Docker version with docker -v. Versions earlier than 19.03 require nvidia-docker2 and the --runtime=nvidia flag. On versions including and after 19.03, you will use the nvidia-container-toolkit package and the --gpus all flag. Both options are documented on the page linked above.


1. Clone this repo:

   ```bash
   git clone https://github.com/rakshithvasudev/jax-world.git 
   ```

2. Build Dockerfile

```bash
cd jax-world/ && docker build -t . jax-world:v1.0
```

3. Run Docker
``` bash 
docker run --rm --gpus=all -v $PWD:/work --ipc=host jax-world:v1.0
```

	

## [Jax](https://github.com/google/jax)
Jax = Numpy + Different kinds of autograd  + H/W Acceleration + XLA/JIT
Lets not forget the functional composition.

## [Optax](https://github.com/deepmind/optax)
Gradient processing and optimization with JAX. Transformations like JAX. Has implementation of :
- Optimizers 
- Loss functions

## [Flax](https://github.com/google/jax#transformations)
Research ready deep learning framework on JAX. Modularity. Supports pytorch like API. Has:
- Neural net APIs
- Optimizers
- Works with large training setups. Accelerated, Multi-node.

## [Haiku](https://github.com/deepmind/dm-haiku)
Another JAX core neural net library. Not a framework.
- Composition
- Tensorflow like with JAX flexibility
- Supports Pytorch like subclassing
- Supports Scale - Distributed training (pmap)
- No optimizers
- Sonnet like programming model
- Ability to inspect and manipulate entires at each layer level
- save internal states whenever needed





