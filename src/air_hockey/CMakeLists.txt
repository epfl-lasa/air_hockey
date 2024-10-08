cmake_minimum_required(VERSION 3.0.2)
project(air_hockey)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  message_generation
)

## Add service files
add_service_files(
  FILES
  Prediction.srv
)

## Add custom messages
add_message_files(
  FILES
  FSM_state.msg
)
generate_messages(
  DEPENDENCIES
  std_msgs
)

## Declare a catkin package
catkin_package(CATKIN_DEPENDS message_runtime roscpp rospy std_msgs)

find_package(Eigen3 REQUIRED eigen3) 
#find_package(osqp REQUIRED)
#find_package(OsqpEigen REQUIRED)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR} "/usr/local/include"
)

add_executable(airhockey src/AirHockey.cpp src/dynamical_system.cpp) 
target_link_libraries(airhockey ${catkin_LIBRARIES} ${Eigen3_LIBRARIES} "stdc++fs" "-L/usr/local/lib")
add_dependencies(airhockey air_hockey_generate_messages_cpp)

add_executable(recorder src/Recorder.cpp)
target_link_libraries(recorder ${catkin_LIBRARIES} ${Eigen3_LIBRARIES} "stdc++fs" "-L/usr/local/lib") 
add_dependencies(recorder air_hockey_generate_messages_cpp)

add_executable(optitrack_transform_publisher src/optitrack_transform_publisher.cpp)
target_link_libraries(optitrack_transform_publisher ${catkin_LIBRARIES})


## Install Python scripts
catkin_install_python(PROGRAMS
  scripts/predict_flux.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
