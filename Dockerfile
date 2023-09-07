FROM python:2.7-slim

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --no-install-suggests -y \
    curl \
    gcc \
    libc-dev \
    imagemagick \
    mencoder \
    parallel \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/*

ENV IS_DOCKER=1
ENV DD_POSE_DIR=/dd-pose
RUN mkdir -p $DD_POSE_DIR
WORKDIR $DD_POSE_DIR

COPY requirements-py27.txt $DD_POSE_DIR

RUN pip install --no-cache-dir -r $DD_POSE_DIR/requirements-py27.txt

COPY 00-activate.sh $DD_POSE_DIR

# make sure bash is the default docker interpreter in order to source our environment
SHELL ["bash", "--login", "-c"]
CMD source 00-activate.sh; bash
