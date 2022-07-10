FROM nvcr.io/nvidia/pytorch:21.03-py3
ARG PYTHON_VERSION=3.8

RUN apt-get update
RUN apt-get install htop -y
RUN apt-get install screen -y
RUN apt-get install psmisc -y

RUN pip install --upgrade pip

RUN pip install \
numpy==1.19 \
tqdm==4.62 \
matplotlib==3.5 \
dill==0.3.4 \
pandas==1.4.1 \
pyarrow==7.0.0 \
nuscenes-devkit==1.1.9 \
l5kit==1.5.0 \
black==22.1.0 \
isort==5.10.1 \
pytest==7.1.1 \
pytest-xdist==2.5.0 \
zarr==2.11.0 \
kornia==0.6.4 

WORKDIR /workspace/trajdata
