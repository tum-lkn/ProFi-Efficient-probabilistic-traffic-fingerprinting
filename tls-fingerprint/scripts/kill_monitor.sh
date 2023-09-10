#!/bin/bash
kill $(ps -e -o pid,command | grep monitor_container | awk 'NR == 1{print$1}')
docker stop $(docker ps | grep grid-search-worker | awk '{ print $1 }')
