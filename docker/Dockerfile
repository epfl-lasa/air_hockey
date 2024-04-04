ARG BASE_IMAGE_TAG=noetic
FROM epfl-lasa/iiwa_ros_double_robot_inertia

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

# iiwa_toolkit
USER root
ARG IIWA_TOOLKIT_BRANCH=feature_ns_inertial_control
WORKDIR /home/${USER}/ros_ws/src
RUN --mount=type=ssh git clone --recursive git@github.com:epfl-lasa/iiwa_toolkit_ns.git --branch ${IIWA_TOOLKIT_BRANCH} 
# feature_ns_inertial_control feature_ns_full_inertia
USER ${USER}

# Python (for python plots)
RUN pip install scipy pandas matplotlib==3.7.4 scikit-learn mplcursors pybullet torch

# Copy i_am_project folder inside docker
WORKDIR /home/${USER}/ros_ws
COPY --chown=${USER} ./ ./src/air-hockey

WORKDIR /home/${USER}/ros_ws
ENV PYTHONPATH "${PYTHONPATH}:/opt/openrobots/lib/python3.8/site-packages/"
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash; catkin_make"
