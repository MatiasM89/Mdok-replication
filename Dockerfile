FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

RUN set -x \
    && apt update \
    && apt install -y git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/mdok

COPY pyproject.toml /opt/mdok/pyproject.toml
RUN mkdir -p mdok && echo 'def main(): pass' > mdok/__init__.py
RUN set -x \
    && python3 -m pip config set global.break-system-packages true \
    && python3 -m pip install --no-cache . \
    && python3 -m pip install --no-cache --no-build-isolation flash-attn \
    && rm -rf ./build ./*.egg-info

COPY mdok/__init__.py /opt/mdok/mdok/__init__.py
RUN python3 -m pip install --no-cache --no-deps . && rm -rf ./build ./*.egg-info

ENV HF_HUB_OFFLINE=1
ENV CUDA_VISIBLE_DEVICES="0"
ENTRYPOINT ["/usr/local/bin/mdok"]
