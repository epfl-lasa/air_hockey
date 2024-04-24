cmake_minimum_required(VERSION 3.0.2)
project(iiwa_toolkit)

add_compile_options(-std=c++17)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  message_generation
  iiwa_tools
  roslib
  dynamic_reconfigure
)

#add dynamic reconfigure api
generate_dynamic_reconfigure_options(
  cfg/passive_cfg_params.cfg
)

find_package(Eigen3 REQUIRED eigen3) 

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES ${PROJECT_NAME}
 CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs message_generation iiwa_tools roslib
#  DEPENDS system_lib
)


include_directories( include ${catkin_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})




add_executable(passive_track src/node_passive_track.cpp  src/passive_control.cpp include/thirdparty/Utils.cpp ${MODEL_SOURCES})
target_link_libraries(passive_track ${catkin_LIBRARIES} ${Eigen3_LIBRARIES})
add_dependencies(passive_track ${PROJECT_NAME}_gencfg)



