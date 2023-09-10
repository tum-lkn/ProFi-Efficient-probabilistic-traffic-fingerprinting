#!/bin/sh
python3 /home/swc/tls-gatherer/src/entrypoint.py 4 0 &
python3 /home/swc/tls-gatherer/src/entrypoint.py 4 1 &
python3 /home/swc/tls-gatherer/src/entrypoint.py 4 2 &
python3 /home/swc/tls-gatherer/src/entrypoint.py 4 3 &

