FROM ubuntu:bionic

RUN apt update --fix-missing -y
RUN apt upgrade -y
RUN apt install iproute2 curl wget gpg software-properties-common firefox chromium-browser -y

RUN curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg
RUN install -o root -g root -m 644 microsoft.gpg /etc/apt/trusted.gpg.d/
RUN sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/edge stable main" > /etc/apt/sources.list.d/microsoft-edge-dev.list'
RUN rm microsoft.gpg
RUN apt-get update
RUN apt-get install microsoft-edge-dev -y

RUN wget -qO- https://deb.opera.com/archive.key | apt-key add
RUN add-apt-repository "deb [arch=i386,amd64] https://deb.opera.com/opera-stable/ stable non-free"
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y opera-stable

RUN apt update --fix-missing -y
RUN DEBIAN_FRONTEND=noninteractive apt install -y tshark


RUN apt install tcpdump tmux htop -y
RUN mv /usr/sbin/tcpdump /usr/bin/tcpdump
RUN apt install ethtool -y
CMD ethtool -K eth0 tx off sg off tso off gro off gso off 	
CMD tail -f /dev/null
