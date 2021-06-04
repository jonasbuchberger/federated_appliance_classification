#!/bin/bash
set -e

MASTER_ADDR="172.24.18.224"
WORLD_SIZE=3
declare -a IP_ARR=("172.24.18.223" "172.24.18.222")

if [[ $1 == "kill" ]]
then
  sudo service docker restart &
  for i in "${!IP_ARR[@]}"
  do
    ssh ubuntu@"${IP_ARR[i]}" "sudo service docker restart" &
  done
else
 sudo docker run -v /home/ubuntu/federated_blond/:/opt/project --network=host --rm --init --ipc=host federated_blond:Dockerfile python3 /opt/project/main_federated.py -r 0 "$WORLD_SIZE" &

 for i in "${!IP_ARR[@]}"
  do
   COMMAND="sudo docker run -v /home/ubuntu/federated_blond/:/opt/project --network=host --rm --init --ipc=host federated_blond:Dockerfile python3 /opt/project/main_federated.py -r $(($i + 1)) -m $MASTER_ADDR $WORLD_SIZE"
   ssh ubuntu@"${IP_ARR[i]}" "$COMMAND" &
  done
fi