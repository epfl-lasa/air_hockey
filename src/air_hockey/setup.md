# Setup for Air Hockey

## Requirements 

The Air Hockey Framework requires several packages to be installed in order to work properly:

* **iiwa_ros**: branch `feature/double_robot_inertia` (and all its dependencies): https://github.com/epfl-lasa/iiwa_ros/tree/feature/inertia
* **iiwa_toolkit**: branch `feature_ns_inertial_control` : https://github.com/epfl-lasa/iiwa_toolkit_ns/tree/feature_ns_inertial_control

It is highly recommended to install everything using [docker](#docker).

## Docker {#docker}

A docker image containing iiwa_ros library is needed to build the air_hockey docker image. 

### Prerequisite

All prerequisite can be installed using the provided install_docker.sh script. 
Details are available [here](https://github.com/epfl-lasa/iiwa_ros/tree/feature/dockerise/docker#prerequisite)

### Docker iiwa-ros
1. Pull the repo 
```bash
git clone -b feature/double-robot-inertia git@github.com:epfl-lasa/iiwa_ros.git
```
    
2. Install docker 
``` bash
cd docker
sudo bash install_docker.sh
```

3. Build the iiwa-ros docker
```bash
bash build_docker.sh
```

### Docker air_hockey
The iiwa_ros docker needs to be built first.

Build air_hockey docker image:
```bash
bash docker/build-image.sh
```

Run docker container:
``` bash 
bash docker/start-docker.sh 
```

Connect form other terminals :
``` bash 
bash docker/start-docker.sh -m connect
```

**Note** : In the start-docker.sh script, volumes are mounted on the folders src/air\_hockey, src/iiwa\_toolkit, python, data. Everything else inside the container and not in these folders will not be mofified on the host machine!

# ROS docker image setup for real robots 

For the real-life setup, we use 2 computers, one connected to each robot. Both need the air_hockey docker image to run this framework.

**IMPORTANT** : Modify the ROS_IP and ROS_MASTER_URI parameters in the build-image.sh script according to the computers IP adresses. Computer 1 is the master.

Computer 1 --> connected to iiwa7 (= iiwa1 = iiwa left)
* ROS_MASTER_URI: 128.178.145.165 
* ROS_IP: 128.178.145.165 
Computer 2 --> connected to iiwa14 (= iiwa2 = iiwa right)
* ROS_MASTER_URI: 128.178.145.165 
* ROS_IP: 128.178.96.118 


# Authors/Maintainers 

Maxime Gautier : maxime.gautier@epfl.ch