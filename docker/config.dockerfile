FROM nvidia/cuda:10.1-cudnn7-runtime

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Setup Ubuntu
RUN apt-get update --yes
RUN apt-get install -y make cmake build-essential autoconf libtool rsync ca-certificates git grep sed dpkg curl wget bzip2 unzip llvm libssl-dev libreadline-dev libncurses5-dev libncursesw5-dev libbz2-dev libsqlite3-dev zlib1g-dev mpich htop vim 

# Get Miniconda and make it the main Python interpreter
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /opt/conda
RUN rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda create -n pytorch_env python=3.6
RUN echo "source activate pytorch_env" > ~/.bashrc
ENV PATH /opt/conda/envs/pytorch_env/bin:$PATH
ENV CONDA_DEFAULT_ENV pytorch_env
RUN conda install pytorch=1.4.0 torchvision cudatoolkit=10.1 -c pytorch
RUN conda install boto3
RUN pip install scipy
RUN git clone https://github.com/google-research/nasbench /code/nasbench
RUN pip install tensorflow==1.12.0
RUN cd /code/nasbench && pip install -e .
RUN pip install matplotlib Cython
RUN pip install ConfigSpace networkx seaborn
RUN pip install xxhash cachetools

RUN mkdir /results
RUN mkdir /nasbench_data

RUN wget https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord -O /nasbench_data/nasbench_only108.tfrecord
RUN pip install awscli


RUN mkdir -p /code/nasbench-1shot1
ADD . /code/nasbench-1shot1
RUN cp code/nasbench-1shot1/cluster_scripts/run-experiment.sh /

