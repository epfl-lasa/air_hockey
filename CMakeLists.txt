cmake_minimum_required(VERSION 3.0.2)
project(i_am_project)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)
find_package(Eigen3 REQUIRED eigen3) 
#find_package(osqp REQUIRED)
#find_package(OsqpEigen REQUIRED)

add_message_files(
  FILES
  FSM_state.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR} "/usr/local/include")

add_executable(air_hockey src/AirHockey.cpp src/dynamical_system.cpp) # src/optimizer.cpp) 
target_link_libraries(air_hockey ${catkin_LIBRARIES} ${Eigen3_LIBRARIES} "stdc++fs" "-L/usr/local/lib") #osqp::osqp OsqpEigen::OsqpEigen
add_dependencies(air_hockey i_am_project_generate_messages_cpp)

add_executable(recorder src/Recorder.cpp)#src/dynamical_system.cpp) # src/optimizer.cpp) 
target_link_libraries(recorder ${catkin_LIBRARIES} ${Eigen3_LIBRARIES} "stdc++fs" "-L/usr/local/lib") #osqp::osqp OsqpEigen::OsqpEigen
add_dependencies(recorder i_am_project_generate_messages_cpp)


