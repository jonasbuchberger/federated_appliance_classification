# Federated Appliance Classification for the BLOND dataset

## Docker
The repo provides two Dockerfiles. One for x86_64 and one for arm_64.
The arm_64 works with the Raspberry Pi4 Ubuntu 64 Bit version.
It is recommended to build the Docker image once and distribute the tar to the other clients.
```shell
# Install Docker client on machine
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Build Docker image
sudo DOCKER-BUILDKIT=1 docker build -t "federated_blond:Dockerfile" .

# Save built Docker environment to tar file
sudo docker save -o c:/image.tar federated_blond:Dockerfile
# Load Docker environment on other client
sudo docker load -i <path to image tar file>
```

## Event Detection
### Preprocess MEDALs
This function is used to preprocess the measurements of MEDALS with the COT feature.
This way, the data can be displayed faster in the GUI.
```python
# src/events/preprocess.py
preprocess_file(storage_path, path_to_data, file, measurement_frequency=6400, net_frequency=50):
```

### Annotator GUI
The Annotator GUI is used for the manual labeling of the BLOND-50 events.
```python
# src/events/annotator_gui.py
Annotator(path_to_data, path_to_log, path_to_preprocessed)

# Start GUI
python3 ~/federated_blond/src/events/annotator_gui.py
```
The GUI can be run local or forwarded over SSH. A X11-Client on the client and server is required.
```shell
# Install client
sudo apt-get install xauth xorg
# Export the display 
export DISPLAY=<CLIENT-IP>:0.0 (VPN-IP)
```

### Event Extraction
The labeled events are further extracted and stored in separated preprocessed files for faster training.
```python
# src/events/extract_events.py
process_event(path_to_data, dest_path, label, measurement_frequency=6400, snippet_length=25600, verbose=False)
```

## Central Models
The central models are trained on the combined data of the BLOND-50.
The experiments include the hyperparameter, feature and architecture search and also the k-fold experiment.
The experiments are started with the **main.py**.
The train config can be changed.
```python
config = {
        'batch_size': 128,                                                          # Number of samples per batch
        'num_epochs': 20,                                                           # Max trained epochs
        'seq_len': 190,                                                             # Number of samples in one event window
        'criterion': torch.nn.CrossEntropyLoss(),                                   # Loss function
        'optim': torch.optim.SGD,                                                   # Optimizer
        'optim_kwargs': {'lr': 0.052, 'weight_decay': 0.001},                       # Optimizer configuration
        'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR,                    # LR scheduler
        'scheduler_kwargs': {'T_max': 50},                                          # Scheduler Configuration
        'model_kwargs': {'name': 'RESNET', 'num_layers': 4, 'start_size': 20},      # Model configuration
        'early_stopping': 50,                                                       # Early stopping patience
        'class_dict': TYPE_CLASS,                                                   # Dictionary of classes to train the model for
        'features': None,                                                           # Features applied on the windows
        'experiment_name': None,                                                    # Folder name of the experiment
        'use_synthetic': True,                                                      # Use the synthetic data
    }
```
```shell
# Run main.py in Docker environment
sudo docker run --name federated_blond --rm -v /home/ubuntu/federated_blond:/opt/project python main.py
```

## Federated Models
The federated experiments can be run in a virtual Docker network or on multiple connected appliances in a local network.
The setup can be modified with a config.
```python
config = {
        'setting': 'noniid',                                                        # iid or non-iid setting
        'batch_size': 128,                                                          # Number of samples per batch    
        'epochs': {'agg_rounds': 100, 'local_steps': 4, 'mode': 'step'},            # Number of epochs, steps and aggregation rounds
        'logging_factor': 1,                                                        # Validation logging per aggregation round
        'seq_len': 190,                                                             # Number of samples in one event window
        'criterion': torch.nn.CrossEntropyLoss(),                                   # Loss function
        'optim': torch.optim.SGD,                                                   # Optimizer
        'optim_kwargs': {'lr': 0.052, 'weight_decay': 0.001},                       # Optimizer configuration
        'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR,                    # LR scheduler
        'scheduler_kwargs': {'T_max': 50},                                          # Scheduler Configuration
        'model_kwargs': {'name': 'RESNET', 'num_layers': 4, 'start_size': 20},      # Model configuration
        'early_stopping': 50,                                                       # Early stopping patience
        'class_dict': TYPE_CLASS,                                                   # Dictionary of classes to train the model for
        'features': None,                                                           # Features applied on the windows
        'experiment_name': None,                                                    # Folder name of the experiment
        'use_synthetic': True,                                                      # Use the synthetic data
        'transfer': False,                                                          # Transfer learning on private data
        'transfer_kwargs': {'lr': 0.075, 'weight_decay': 0.0, 'num_epochs': 10},    # Transfer learning parameters
        'local_test': False,                                                        # Test global model on each client
        'weighted': False,                                                          # Personalized model weighting FedFomo
    }
```

### Federated Docker Environment
In order to run the federated experiment in the federated Docker environment, a virtual network has to be created.
Then the experiment can be started.

- Rank: 0 Server, 1-n for clients
- World Size: Number of clients + server

```shell
# Create virtual Docker network for federated Docker environment
sudo docker network create --subnet=10.18.0.0/16 fednet

# Run single client
sudo docker run -v /home/ubuntu/federated_blond/:/opt/project --rm --init --ipc=host --network=fednet --ip=10.18.0.<IP> federated_blond:Dockerfile python3 /opt/project/main_federated.py -r <Rank> -m 10.18.0.50 <World Size> &
# Run all client and server automatically, also creates network
./run.sh local
```
### Federated Client Environment
The first step is to distribute the python files for the client and dataset classes to all clients.
This functionality is supplied with the copy.sh.
The IP addresses need to be adjusted.
For the experiments the data also needs to be distributed to each client once.
```shell
# Copy files to clients
./copy.sh files
# Copy files to clients
./copy.sh data
```
After the files is distributed the server starts the clients over SSH.
```shell
# Start local server
sudo docker run -v /home/ubuntu/federated_blond/:/opt/project --network=host --rm --init --ipc=host federated_blond:Dockerfile python3 /opt/project/main_federated.py -r 0 <World Size> &
# Start one client over SSH
ssh ubuntu@<Client IP> "sudo docker run -v /home/ubuntu/federated_blond/:/opt/project --network=host --rm --init --ipc=host federated_blond:Dockerfile python3 /opt/project/main_federated.py -r <Rank> -m <Server IP> <World Size> &
```
On the server the automatic run script can be used **run.sh**.
The IP adresses of the clients need to be adjusted.
```shell
# Start clients and server
./run.sh
# Kill all Docker runs
./run.sh kill
```