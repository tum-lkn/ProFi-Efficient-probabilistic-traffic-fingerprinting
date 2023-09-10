FROM python:3.9-buster
LABEL maintainer n.landerer@tum.de

COPY requirements.txt /usr/bin
RUN pip3 install -U -r /usr/bin/requirements.txt

ENTRYPOINT python3 /mnt/host/task_scheduler.py