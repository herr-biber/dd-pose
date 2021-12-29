FROM ubuntu:20.04

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    parallel \
    curl \
    imagemagick \
    mencoder \
    && rm -rf /var/lib/apt/lists/*

ENV IS_DOCKER=1
ENV DD_POSE_DIR=/dd-pose
RUN mkdir -p $DD_POSE_DIR
WORKDIR $DD_POSE_DIR

COPY requirements.txt $DD_POSE_DIR

RUN pip3 install --no-cache-dir pip>=19.3 --upgrade
RUN pip3 install --no-cache-dir -r $DD_POSE_DIR/requirements.txt --upgrade

COPY 00-activate.sh $DD_POSE_DIR

# make sure bash is the default docker interpreter in order to source our environment
SHELL ["bash", "--login", "-c"]
CMD source 00-activate.sh; bash
