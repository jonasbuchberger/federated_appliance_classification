FROM ubuntu:20.04

RUN apt-get update && \
    apt-get upgrade --yes

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install --yes software-properties-common python3 python3-pip && \
    apt-get install --yes git && \
    apt-get install --yes libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config ffmpeg && \
    apt-get install --yes libvpx-dev libopus-dev libffi-dev

# Install torch
RUN pip3 install torch==1.8.1
RUN pip3 install torchvision==0.9.1
RUN pip3 install torchaudio -f https://torch.kmtea.eu/whl/stable.html

RUN apt-get install llvm --yes
RUN pip3 install librosa

RUN apt-get install --yes libhdf5-103 libhdf5-dev
RUN pip3 install h5py
RUN pip3 install tensorboard
RUN pip3 install tensorboardX
RUN pip3 install smt
RUN pip3 install tqdm
RUN pip3 install scikit-learn
RUN pip3 install pandas
