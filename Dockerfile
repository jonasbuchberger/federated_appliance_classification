FROM ubuntu:18.04
# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    git \
    bzip2 \
    libx11-6 \
    tmux \
    gcc \
    g++ \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p /home/ubuntu/miniconda \
 && rm ~/miniconda.sh

ENV PATH=/home/ubuntu/miniconda/bin:$PATH
RUN conda clean -ya

RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch
RUN conda install matplotlib
RUN conda install scikit-learn
RUN conda install pandas
RUN conda install -c conda-forge notebook
RUN conda install h5py
RUN conda clean -ya

RUN pip install tensorboard
RUN pip install tensorboardX
RUN pip install syft
RUN pip install smt

# Set the default command to python3
CMD ["python3"]
