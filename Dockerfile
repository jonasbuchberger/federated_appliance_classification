FROM ubuntu:18.04
# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    tmux \
    # Necessary for latest tune patch
    gcc \         
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /workingdir
WORKDIR /workingdir

# Create a non-root user and switch to it
RUN adduser --uid 1002 --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /workingdir
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.9
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda clean -ya

# CUDA 10.1-specific steps
RUN conda install -y -c pytorch torchvision torchaudio cpuonly -c pytorch \
    matplotlib \
    scikit-learn \
    pandas \
    jupyter \
    && conda clean -ya

RUN pip install tensorboard
RUN pip install tensorboardX
RUN pip install syft
# Set the default command to python3
CMD ["python3"]
