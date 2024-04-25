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

# install development dependencies and copy licenses
COPY --from=ghcr.io/aica-technology/control-libraries/development-dependencies:20.04 /usr/local /tmp/local
RUN sudo cp -R /tmp/local/* /usr/local && sudo rm -r /tmp/local
COPY --from=ghcr.io/aica-technology/control-libraries/development-dependencies:20.04 \
  /usr/share/doc/control-libraries/licenses /usr/share/doc/control-libraries/licenses
  
FROM install_stage AS control-libraries-build
WORKDIR ${HOME}

RUN sudo ldconfig

ARG CL_BRANCH=main
RUN git clone -b ${CL_BRANCH} --depth 1 https://github.com/aica-technology/control-libraries.git

# install source
RUN sudo ./control-libraries/source/install.sh -y

# install protocol
RUN sudo ./control-libraries/protocol/install.sh -y

# install python bindings
#RUN sudo pip3 install ./control-libraries/python

# clean image
RUN sudo rm -rf ./control-libraries && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

FROM install_stage AS copy_cl_stage
RUN sudo rm -rf /usr/include/eigen3
COPY --from=control-libraries-build /usr/ /usr/
RUN sudo ldconfig

# Copy src, data, python and docker folders inside docker
FROM copy_cl_stage AS copy_stage
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
