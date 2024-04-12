# Air_hockey 

This repository contains the complete code for the data collection framework dubbed "Air Hockey". This framework consists of two KUKA IIWA robots hitting an object back and forth between each other's workspace. The hit object slides on a surface, leaving the hitting robot's workspace to end up in the receiving robot's workspace. The roles are then reversed and the object is hit back in the opposite direction. This setup allows for fast data colelction on hitting objects, sliding and impact.


TODO : 
- add picture + diagram to READ ME
- make setup.md
- cite Harhsit's paper


## Requirements
airhockey requires several packages to be installed in order to work properly:

It is recommended to install everything using docker (cf. docker section)

* **iiwa_ros**: branch `feature/double_robot_inertia` (and all its dependencies): https://github.com/epfl-lasa/iiwa_ros/tree/feature/inertia

## Docker

A docker containing iiwa-ros library is needed to build the air_hockey docker.

### Prerequisite

cf. https://github.com/epfl-lasa/iiwa_ros/tree/feature/dockerise/docker#prerequisite

### Docker iiwa-ros
1. Pull the repo 
```bash
git clone -b feature/double-robot-inertia git@github.com:epfl-lasa/iiwa_ros.git
```
    
2. Build the iiwa-ros docker
``` bash
cd docker
bash install_docker.sh
```

### Docker air_hockey
The iiwa-ros docker needs to build first.

Build docker:
The branch of iiwa-toolkit lib can be chosen. The default branch is feature_inertial_control
```bash
./docker/build-server.sh
```

Run docker:
``` bash 
bash docker/start-docker.sh 
```
Connect form other terminals :
``` bash 
bash docker/start-docker.sh -m connect
```

Note : volumes are mounted on the folders src/air\_hockey, src/iiwa\_toolkit, python, data and docker. Everything inside the container not in these folders will not be mofified on the host machine!


# Run the simulation

Run thedocker container, then in one terminal, launch the gazebo simulation:
``` bash
roslaunch air_hockey airhockey_sim.launch
```

# Real setup

``` bash 
aica-docker interactive air_hockey:noetic -u ros --net host --no-hostname -v /home/maxime/Documents/air_hockey:/home/ros/ros_ws/src/air_hockey --privileged
```

Computer 1 - IP: 128.178.145.165 -> connected to iiwa7 (= iiwa1 = iiwa left)
Computer 2 - IP: 128.178.96.118 -> connected to iiwa14(= iiwa2 = iiwa right)

Connect via ssh from computer 1 to computer 2 : 
```bash
ssh elise@128.178.96.118
```

Launch docker on both computers with aica-docker command above or using the alias :
```bash
airhockey
```

Connect with other terminals using : 
```bash
aica-docker connect air_hockey-noetic-runtime -u ros
```
(use this command on computer 2 to leave container running)

You should have 4 terminals, 3 connected to the docker container on computer 1 and one on the docker container on Computer 2. From now on, Computer 2 will refer to that last terminal (connected via ssh)

Don't forget to run catkin_make to build the latest version of the code!!!

## Setup ROS communications

Modify ROS_MASTER_URI and ROS_IP inside both containers for proper ROS communications

both : 
```bash
sudo nano ~/.bashrc
```
Computer 1 : 
```bash
export ROS_MASTER_URI=http://128.178.145.165:11311 
export ROS_IP=128.178.145.165
```
Computer 2 : 
```bash
export ROS_MASTER_URI=http://128.178.145.165:11311 
export ROS_IP=128.178.96.118
```
both : 
```bash
source ~/.bashrc
```
## ROS launch commands

Launch the following commands in this order

Computer 1 - Terminal 1 :
```bash
roslaunch air_hockey optitrack_with_transform_to_base.launch
```
Computer 2 - Terminal 2 :
```bash
roslaunch iiwa_toolkit passive_track_real.launch robot_name:=iiwa2 model:=14
```
Computer 1 - Terminal 3 :
```bash
roslaunch iiwa_toolkit passive_track_real.launch robot_name:=iiwa1 model:=7
```
Computer 1 - Terminal 4 :
```bash
roslaunch air_hockey air_hockey_real.launch
```

Sequence for iiwa trackpad :
iiwa7 : FRIOverlay -> Torque, 150, 0.6 (50,0.5)
iiwa14 : FRIOverlay -> Torque, 300, 0.2 

### Remarks and troubleshooting

Both computers must have the IP adress 190.170.10.1 for the conenction to the iiwa as FRIOverlay communicates only with this adress.
Can modify the IP adress of the robot in iiwa_driver/config/iiwa.yaml

For optitrack, make sure Motive is streaming in Y-up format, and that all object frames matches the robot frame (X in front, Y to the left and Z up)

Use 'sudo chmod 777 -R .' when having trouble saving files inside container 

Use 'git config core.fileMode false' to avoid pushing chmod changes to github


# Authors/Maintainers 

Maxime Gautier : maxime.gautier@epfl.ch
