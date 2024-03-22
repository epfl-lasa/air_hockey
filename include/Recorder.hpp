//|
//|    Copyright (C) 2021-2023 Learning Algorithms and Systems Laboratory, EPFL, Switzerland
//|    Authors: Harshit Khurana (maintainer)
//|
//|    email:   harshit.khurana@epfl.ch
//|
//|    website: lasa.epfl.ch
//|
//|    This file is part of iam_dual_arm_control.
//|    This work was supported by the European Community's Horizon 2020 Research and Innovation
//|    programme (call: H2020-ICT-09-2019-2020, RIA), grant agreement 871899 Impact-Aware Manipulation.
//|
//|    iam_dual_arm_control is free software: you can redistribute it and/or modify  it under the terms
//|    of the GNU General Public License as published by  the Free Software Foundation,
//|    either version 3 of the License, or  (at your option) any later version.
//|
//|    iam_dual_arm_control is distributed in the hope that it will be useful,
//|    but WITHOUT ANY WARRANTY; without even the implied warranty of
//|    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//|    GNU General Public License for more details.
//|

#pragma once

#include <ros/ros.h>
#include <ros/package.h>
#include <ros/console.h>

#include <gazebo_msgs/LinkStates.h>
#include <gazebo_msgs/ModelStates.h>
#include <geometry_msgs/Inertia.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <sys/stat.h>
#include <string>
#include <iomanip>
#include <limits>

#include "i_am_project/FSM_state.h"

#define NB_ROBOTS 2// Number of robots

class Recorder {
private:
  enum Robot { IIWA_7 = 0, IIWA_14 = 1 };
  enum robotMode{REST, HIT};

  struct FSMState{
    robotMode mode_iiwa7 = REST;
    robotMode mode_iiwa14 = REST;
    bool isHit = 0;
    ros::Time hit_time = ros::Time::now();
    float des_flux;
    Eigen::Vector3f des_pos;
  };

  struct RecordedRobotState {
    std::string robot_name;
    ros::Time time;
    Eigen::VectorXd joint_pos = Eigen::VectorXd(7);
    Eigen::VectorXd joint_vel = Eigen::VectorXd(7);
    Eigen::VectorXd joint_effort = Eigen::VectorXd(7);
    Eigen::VectorXd trq_cmd = Eigen::VectorXd(7);
    Eigen::Vector3f eef_pos;
    Eigen::Vector4f eef_orientation;
    Eigen::Vector3f eef_vel;
    Eigen::Vector3f eef_vel_des;
    Eigen::Matrix<float, 9, 1> inertia;
    float hitting_flux;
  };

  struct RecordedObjectState {
    ros::Time time;
    Eigen::Vector3f position;
    Eigen::Vector4f orientation;
  };

  bool isSim_;
  bool isRecording_;
  bool isFluxFixed_;

  FSMState fsmState_;

  std::string recordingFolderPath_;
  std::string fluxFilename_;
  float recordingTimeObject_;
  float recordingTimeRobot_;

  float objectMass_;
  float hittingFluxDes_[NB_ROBOTS];
  ros::Time hittingTime_[NB_ROBOTS];
  Eigen::Vector3f desiredPosition_[NB_ROBOTS];

  std::string pubVelQuatTopic_[NB_ROBOTS];
  std::string iiwaInertiaTopic_[NB_ROBOTS];
  std::string iiwaPositionTopicSim_;
  std::string objectPositionTopic_;
  std::string iiwaPositionTopicReal_[NB_ROBOTS];
  std::string iiwaVelocityTopicReal_[NB_ROBOTS];
  std::string iiwaJointStateTopicReal_[NB_ROBOTS];
  std::string iiwaTorqueCmdTopic_[NB_ROBOTS];
  std::string FSMTopic_;

  ros::Rate rate_;
  ros::NodeHandle nh_;
  ros::Publisher pubVelQuat_[NB_ROBOTS];
  ros::Subscriber objectPosition_;
  ros::Subscriber iiwaPosition_;
  ros::Subscriber iiwaInertia_[NB_ROBOTS];
  ros::Subscriber iiwaPositionReal_[NB_ROBOTS];
  ros::Subscriber iiwaVelocityReal_[NB_ROBOTS];
  ros::Subscriber iiwaJointStateReal_[NB_ROBOTS];
  ros::Subscriber iiwaDesiredVelocity_[NB_ROBOTS];
  ros::Subscriber iiwaTorqueCmd_[NB_ROBOTS];
  ros::Subscriber FSMState_;

  geometry_msgs::Pose boxPose_;
  geometry_msgs::Pose iiwaPose_[NB_ROBOTS];
  geometry_msgs::Twist iiwaVel_[NB_ROBOTS];
  sensor_msgs::JointState iiwaJointState_[NB_ROBOTS];

  Eigen::Vector3f objectPositionFromSource_;
  Eigen::Vector3f previousObjectPositionFromSource_;
  Eigen::Vector4f objectOrientationFromSource_;
  Eigen::Vector3f iiwaPositionFromSource_[NB_ROBOTS];
  Eigen::Vector4f iiwaOrientationFromSource_[NB_ROBOTS];
  Eigen::Vector3f iiwaVelocityFromSource_[NB_ROBOTS];
  Eigen::Matrix3f iiwaTaskInertiaPosInv_[NB_ROBOTS];;
  Eigen::Vector3f iiwaDesiredVelocityFromSource_[NB_ROBOTS];
  Eigen::VectorXd iiwaTorqueCmdFromSource_[NB_ROBOTS];
  bool isObjectMoving_;
  int moved_manually_count_;

  std::vector<RecordedRobotState> robotStatesVector_[NB_ROBOTS];
  std::vector<RecordedObjectState> objectStatesVector_;
  std::vector<RecordedObjectState> objectStatesVectorManual_;

public:

  explicit Recorder(ros::NodeHandle& nh, float frequency) : nh_(nh), rate_(frequency){};

  bool init();

  void run();

  int getIndex(std::vector<std::string> v, std::string value);
  void iiwaInertiaCallback(const geometry_msgs::Inertia::ConstPtr& msg, int k);
  void iiwaPositionCallbackGazebo(const gazebo_msgs::LinkStates& linkStates);
  void objectPositionCallbackGazebo(const gazebo_msgs::ModelStates& modelStates);
  void iiwaJointStateCallbackReal(const sensor_msgs::JointState::ConstPtr& msg, int k);
  void iiwaDesiredVelocityCallback(const geometry_msgs::Pose::ConstPtr& msg, int k);

  void iiwaPoseCallbackReal(const geometry_msgs::Pose::ConstPtr& msg, int k);
  void iiwaVelocityCallbackReal(const geometry_msgs::Twist::ConstPtr& msg, int k);
  void iiwaBasePositionCallbackReal(const geometry_msgs::PoseStamped::ConstPtr& msg, int k);
  void objectPositionCallbackReal(const geometry_msgs::PoseStamped::ConstPtr& msg);
  void iiwaTorqueCmdCallback(const std_msgs::Float64MultiArray::ConstPtr &msg, int k);
  void FSMCallback(const i_am_project::FSM_state::ConstPtr& msg);

  void recordRobot(Robot robot_name);
  void recordObject(bool manual);
  void recordObjectMovedByHand(int hit_count); 
  void writeRobotStatesToFile(Robot robot_name, int hit_count);
  void writeObjectStatesToFile(int hit_count, std::string filename, bool manual);
  void copyYamlFile(std::string inFilePath, std::string outFilePath);
  void setUpRecordingDir();
  std::string robotToString(Robot robot_name);
  float calculateDirFlux(Robot robot_name);
};