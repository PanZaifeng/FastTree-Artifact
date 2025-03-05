FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

RUN apt-get update
RUN apt-get install -y curl git python3.10 python3.10-dev python3.10-distutils

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.10 get-pip.py
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN pip install --upgrade pip
RUN pip install "sglang[all]==0.2.13"
RUN pip install flash-attn==2.6.3
RUN pip install flashinfer==0.1.6 -i https://flashinfer.ai/whl/cu121/torch2.4

RUN pip install matplotlib==3.9.2
RUN pip install pandas==2.2.3
