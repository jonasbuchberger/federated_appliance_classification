#!/bin/bash

if [[ -z $1 ]]; then
    echo -e "Usage: ./run_docker.sh <file to execute.(py | ipynb)>\nor ./run_docker.sh bash for a shell\nor ./run_docker background <name> for starting a container in the background\nor ./run_docker.sh tensorboard for starting a tensorboard server"
    exit 1
fi 

# Create log dir if not already exists
! [[ -d logs ]] && mkdir logs
# Name of logfile
log_name=`echo "$(date +%d-%m-%Y-%T)-$1" | sed -r 's/\/+/_/g'`

# Base directory for additional folders on new machine
base="/home/ubuntu/federated_blond"
# Base directory for additional folders on old machine
#DATA_DIR_MOUNT="$base/data:/workingdir/data"
#RUNS_DIR_MOUNT="$base/runs:/workingdir/runs"
CWD_DIR_MOUNT="$PWD:/workingdir"

BASE_DOCKER_RUN_CMD_ARGS="--rm --init -it --ipc=host --user="0:0" --volume $CWD_DIR_MOUNT federated_blond" # --volume $DATA_DIR_MOUNT --volume $RUNS_DIR_MOUNT

# Check for file extension
if [[ $1 =~ \.ipynb ]]; then 
    docker run $BASE_DOCKER_RUN_CMD_ARGS jupyter nbconvert --to notebook --inplace --execute $1 2>&1 > logs/$log_name.log 
elif [[ $1 =~ \.py ]]; then 
    docker run $BASE_DOCKER_RUN_CMD_ARGS python3 $1 2>&1 > logs/$log_name.log 
elif [[ $1 == "bash" ]]; then
    docker run $BASE_DOCKER_RUN_CMD_ARGS bash
    exit 0  
elif [[ $1 == "background" ]]; then 
    docker run -d  --name idpradio2_$2 $BASE_DOCKER_RUN_CMD_ARGS bash
    exit 0 
elif [[ $1 == "tensorboard" ]]; then
    docker run -d --rm --init -it --name federated_blond_tensorboard --user="0:0" -p 9000:6006 --volume $RUNS_DIR_MOUNT federated_blond tensorboard --logdir /workingdir/runs --host 0.0.0.0
    exit 0
else
    echo "$1 is not supported"
    exit 1
fi 
# Retrieve exit code of docker command
EXIT_CODE=$?
# Remove colors from log file
sed -i -r "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[m|K]//g" ./logs/$log_name.log
# If docker command was succesfull, delete logfile
[ $EXIT_CODE -eq 0 ] && rm ./logs/$log_name.log
