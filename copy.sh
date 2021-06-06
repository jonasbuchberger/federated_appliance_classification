#!/bin/bash
set -e

BASE_DIR="/home/ubuntu"
declare -a IP_ARR=("172.24.18.222" "172.24.18.223" "172.24.18.224" "172.24.18.246" "172.24.18.247" "172.24.18.248")


if [[ $1 == "files" ]]
then
  for i in "${!IP_ARR[@]}"
  do
    rsync --include={src/data,src/models} --exclude={data,models,notebooks,.old,__pycache__,.idea,.git} -av -e "ssh -i ~/.ssh/id_rsa" "$BASE_DIR/federated_blond" ubuntu@"${IP_ARR[i]}":~ --delete
  done
elif [[ $1 == "data" ]]
then
  for i in "${!IP_ARR[@]}"
  do
    rsync -av -e "ssh -i ~/.ssh/id_rsa" "$BASE_DIR/federated_blond/data" ubuntu@"${IP_ARR[i]}":~/federated_blond/ --delete
  done
fi
