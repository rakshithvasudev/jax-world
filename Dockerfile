FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04



#RUN rm /etc/apt/sources.list.d/cuda.list \
# && apt update \
# && apt install -y --no-install-recommends build-essential autoconf libtool git \
#        ccache curl wget pkg-config sudo ca-certificates automake libssl-dev \
#        bc python3-dev python3-pip google-perftools gdb libglib2.0-dev clang sshfs libre2-dev \
#        libboost-dev libnuma-dev numactl sysstat sshpass ntpdate less vim iputils-ping \


# install python3-pip
RUN apt update && apt install vim python3-pip -y


RUN pip3 install --upgrade pip

# install dependencies via pip
RUN pip3 install numpy scipy six wheel jupyterlab

RUN pip3 install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

RUN pip3 install --upgrade flax optax tensorflow_datasets torch torchvision dm-haiku transformers 



