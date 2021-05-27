#!/bin/bash
set -e

BASE_DIR="/mnt/c/Users/jonas/Documents/PyCharm"
MASTER_ADDR="192.168.178.20"
WORLD_SIZE=2
declare -a IP_ARR=("192.168.178.20")


if [[ $1 == "files" ]]
then
  for i in "${!IP_ARR[@]}"
  do
    rsync --include={src/data,src/models} --exclude={data,models,notebooks,.old,__pycache__,.idea,.git} -av -e "ssh -i ~/.ssh/id_rsa" "$BASE_DIR/federated_blond" ubuntu@"${IP_ARR[i]}":~
  done
elif [[ $1 == "data" ]]
then
  for i in "${!IP_ARR[@]}"
  do
    rsync -av -e "ssh -i ~/.ssh/id_rsa" "$BASE_DIR/federated_blond/data" ubuntu@"${IP_ARR[i]}":~/federated_blond/
  done
elif [[ $1 == "run" ]]
  sudo docker run -v /home/ubuntu/:/opt/project --network=host --rm --init --ipc=host -it federated_blond:Dockerfile python3 /opt/project/worker.py -r 0 "$WORLD_SIZE" &
then
  for i in "${!IP_ARR[@]}"
  do
     COMMAND="sudo docker run -v /home/ubuntu/:/opt/project --network=host --rm --init --ipc=host federated_blond:Dockerfile python3 /opt/project/worker.py -r $(($i + 1)) -m $MASTER_ADDR $WORLD_SIZE"
     ssh -i ~/.ssh/id_rsa ubuntu@"${IP_ARR[i]}" "$COMMAND" &
  done
fi
