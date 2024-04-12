# Air_hockey 

This package contains the high-level control for the Air Hockey framework. It consists of :
- a Finite State Machine node handling the hitting state of the robots, which can be used autonomously or with keyboard input.
- a recorder node handling the recording of data when hitting. It records both the object and robot state.
- an optitrack node to transform the raw optitrack data to the base of each robot for cleaner data collection


TODO : add airhockey logic digram 

## Requirements

The docker image should be built according to the [setup instructions](setup.md).

## Usage

See [usage file](src/air_hockey/usage.md) for detailed instructions on using the framework.

For a quick working test, you can run the gazebo simulation using:
``` bash
cd air_hockey
bash docker/start-docker.sh
roslaunch air_hockey air_hockey_sim.launch
```

# Authors/Maintainers 

Maxime Gautier : maxime.gautier@epfl.ch
