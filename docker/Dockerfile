ARG BASE_IMAGE_TAG=noetic
FROM epfl-lasa/iiwa_ros_double_robot_inertia AS install_stage

# Modify files inside docker
ARG HOST_GID=1001
ENV USER_GROUP=${USER}
USER root
RUN if [ "${HOST_GID}" != "1000" ];\
    then groupadd --gid ${HOST_GID} host_group && \
    usermod ${USER} -g ${HOST_GID} && \ 
    usermod ${USER} -a -G ${USER_GROUP}; fi
USER ${USER}

# Add vrpn tools
RUN sudo apt-get update && sudo apt-get install -y \
    netbase \
    ros-noetic-vrpn-client-ros \
    ros-noetic-rqt-reconfigure \
    ros-noetic-rosbridge-server \
    && sudo apt-get upgrade -y && sudo apt-get clean

# Python (for python plots)
RUN pip install scipy pandas matplotlib==3.7.4 scikit-learn mplcursors pybullet torch gmr

# Copy src, data, python and docker folders inside docker
FROM install_stage AS copy_stage
WORKDIR /home/${USER}/ros_ws
COPY --chown=${USER} ./data ./data/
COPY --chown=${USER} ./python ./python/
COPY --chown=${USER} ./src ./src/

# Build catkin workspace
FROM copy_stage AS build_stage
WORKDIR /home/${USER}/ros_ws
ENV PYTHONPATH "${PYTHONPATH}:/opt/openrobots/lib/python3.8/site-packages/"
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash; catkin_make"

# Set ROS IP for communciation in real setup
FROM build_stage AS ros_stage
ARG ROS_IP
ARG ROS_MASTER_IP

ENV ROS_IP=${ROS_IP}
ENV ROS_MASTER_URI=http://${ROS_MASTER_IP}:11311

RUN echo "export ROS_MASTER_URI=http://${ROS_MASTER_IP}:11311" >> /home/${USER}/.bashrc;
RUN echo "export ROS_IP=${ROS_IP}" >> /home/${USER}/.bashrc ;