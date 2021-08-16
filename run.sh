#!/bin/bash
set -e

declare -a IP_ARR=("172.24.18.217" "172.24.18.218" "172.24.18.219" "172.24.18.220" "172.24.18.221" "172.24.18.222" "172.24.18.223" "172.24.18.241" "172.24.18.242" "172.24.18.243" "172.24.18.244" "172.24.18.245" "172.24.18.246" "172.24.18.247" "172.24.18.248")
MASTER_ADDR="172.24.18.224"
WORLD_SIZE=${#IP_ARR[@]}
WORLD_SIZE=$(($WORLD_SIZE + 1))

if [[ $1 == "kill" ]]
then
  sudo service docker restart &
  for i in "${!IP_ARR[@]}"
  do
    ssh ubuntu@"${IP_ARR[i]}" "sudo service docker restart" &
  done
elif [[ $1 == "local" ]]
then
    sudo docker network rm fednet
    sudo docker network create --subnet=10.18.0.0/16 fednet
  for i in "${!IP_ARR[@]}"
  do
    sudo docker run -v /home/ubuntu/federated_blond/:/opt/project --rm --init --ipc=host --network=fednet --ip=10.18.0.$(($i + 51)) federated_blond:Dockerfile python3 /opt/project/main_federated.py -r $(($i + 1)) -m 10.18.0.50 $WORLD_SIZE &
  done
  sudo docker run -v /home/ubuntu/federated_blond/:/opt/project --rm --init --ipc=host --network=fednet --ip=10.18.0.50 federated_blond:Dockerfile python3 /opt/project/main_federated.py -r 0 "$WORLD_SIZE"
else
 sudo docker run -v /home/ubuntu/federated_blond/:/opt/project --network=host --rm --init --ipc=host federated_blond:Dockerfile python3 /opt/project/main_federated.py -r 0 "$WORLD_SIZE" &

 for i in "${!IP_ARR[@]}"
  do
   COMMAND="sudo docker run -v /home/ubuntu/federated_blond/:/opt/project --network=host --rm --init --ipc=host federated_blond:Dockerfile python3 /opt/project/main_federated.py -r $(($i + 1)) -m $MASTER_ADDR $WORLD_SIZE"
   ssh ubuntu@"${IP_ARR[i]}" "$COMMAND" &
  done
fi