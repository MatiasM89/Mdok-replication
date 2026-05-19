FROM registry.webis.de/code-research/tira/tira-user-mdok/mdok-eadc2:latest-tira-docker-software-id-caramel-twig

RUN set -x \
    && apt update \
    && apt install -y git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && rm -Rf /opt/mdok

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

RUN python3 -m pip uninstall -y transformers \
	&& python3 -m pip install transformers==5.8.1 \
	&& python3 -m pip install -U bitsandbytes>=0.46.1

ENV HF_HUB_OFFLINE=1
ENTRYPOINT ["/usr/local/bin/mdok"]
