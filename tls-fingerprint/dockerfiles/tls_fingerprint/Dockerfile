FROM ubuntu:20.04

USER root

RUN apt-get update \
    && apt-get -yq upgrade \
    # && apt install software-properties-common \
    # && add-apt-repository ppa:wireshark-dev/stable \
    && apt-get update \
    && apt install -yq tshark \
    && DEBIAN_FRONTEND=noninteractive apt-get install -qqy graphviz libgraphviz-dev graphviz-dev pkg-config \
    && apt-get install -y python3-pip python3-dev vim openssh-server sudo\
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python \
    && pip3 install --upgrade pip \
    && python -m pip install numpy pandas matplotlib jupyter \
    && python -m pip install networkx \
    && python -m pip install SQLAlchemy \
    && python -m pip install PyMySQL \
    && python -m pip install tables \
    && python -m pip install h5py \
    && python -m pip install scikit-learn \
    && python -m pip install torch \
    && python -m pip install pyro-ppl \
    && python -m pip install ray \
    && python -m pip install ray[tune] \
    && python -m pip install pygraphviz \
    && python -m pip install python-Levenshtein

RUN pip install docker

RUN apt-get update && apt-get install libpcap-dev -y

RUN service ssh start
RUN useradd -rm -d /home/developer -s /bin/bash -g root -G sudo -u 1000 developer
RUN  echo 'developer:developer' | chpasswd
EXPOSE 22222

# RUN export uid=1000 gid=1000 && \
#     mkdir -p /home/developer && \
#     # mkdir -p /etc/sudoers.d && \
#     echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
#     echo "developer:x:${uid}:" >> /etc/group && \
#     echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
#     chmod 0440 /etc/sudoers.d/developer && \
#     chown ${uid}:${gid} -R /home/developer
USER developer
ENV HOME /home/developer
WORKDIR /home/developer

