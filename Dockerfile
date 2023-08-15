FROM ubuntu:22.04

ARG BOOST_VERSION=1.83.0

###################
### IMAGE SETUP ###
###################

RUN apt-get update && \
    apt-get install -y \
    git \
    g++ \
    wget \
    bzip2 \
    cmake \
    libsuitesparse-dev

#######################
### DIRECTORY SETUP ###
#######################

RUN mkdir /amdis

#####################
### INSTALL BOOST ###
#####################

# install script for boost taken from:
# https://leimao.github.io/blog/Boost-Docker/

# all commands below need to be concatinated with && so the
# BOOST_VERSION_MOD variable can be used throughout. Setting
# it in one layer will not carry over to the other layers.

RUN cd /tmp && \
    BOOST_VERSION_MOD=$(echo $BOOST_VERSION | tr . _) && \
    wget https://boostorg.jfrog.io/artifactory/main/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_MOD}.tar.bz2 && \
    tar --bzip2 -xf boost_${BOOST_VERSION_MOD}.tar.bz2 && \
    cd boost_${BOOST_VERSION_MOD} && \
    ./bootstrap.sh --prefix=/usr/local && \
    ./b2 install && \
    rm -rf /tmp/*

#####################
### INSTALL AMDIS ###
#####################

RUN cd /
# need to clone via https, otherwise there is an authentication failure
RUN git clone https://gitlab.mn.tu-dresden.de/iwr/amdis.git amdis_repo
RUN cd amdis_repo

RUN mkdir build && cd build
RUN cmake /amdis_repo/AMDiS

RUN cmake -DCMAKE_INSTALL_PREFIX:PATH=/amdis .
RUN cmake --build . --target install

#####################
### PROJECT SETUP ###
#####################

##################
### ENTRYPOINT ###
##################
