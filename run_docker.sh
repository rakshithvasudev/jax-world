#!/bin/bash

docker run --rm -it -v $PWD:/work \
	--gpus=all --ulimit memlock=-1 \
	--ulimit stack=67108864 \
	--net=host --ipc=host \
 	--workdir /work	\
	jax-world:v1.0 bash 
