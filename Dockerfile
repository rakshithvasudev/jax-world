FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

# install python3-pip
RUN apt-get update && apt-get install python3-pip -y


RUN pip3 install --upgrade pip

# install dependencies via pip
RUN pip3 install numpy scipy six wheel jupyterlab

RUN pip3 install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html



