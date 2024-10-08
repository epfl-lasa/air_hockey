//|
//|    Copyright (C) 2021-2023 Learning Algorithms and Systems Laboratory, EPFL, Switzerland
//|    Authors: Harshit Khurana (maintainer)
//|    	        Maxime Gautier (maintainer)
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

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <sys/stat.h>
#include <string>
#include <iomanip>
#include <limits>
#include <sstream>

#include "dynamical_system.h"
#include "keyboard_interaction.hpp"
#include "air_hockey/FSM_state.h"
#include "air_hockey/Prediction.h"

#define NB_ROBOTS 2// Number of robots

class AirHockey {
private:
  enum Robot { IIWA_7 = 0, IIWA_14 = 1, NONE = 2 };
  enum robotMode{REST, HIT};

  struct FSMState{
    robotMode mode_iiwa7 = REST;
    robotMode mode_iiwa14 = REST;
    bool isHit = 0;
    ros::Time hit_time = ros::Time::now();
    float des_flux;
    Eigen::Vector3f des_pos;
  };

  bool isHit_ = 0;
  bool isSim_;
  bool isAuto_;
  bool isAiming_;
  bool useMovingTarget_;
  bool isPaused_;
  bool isFluxFixed_;
  bool isObjectMoving_;
  bool callFluxService_ = true;
  float timeToWait_;
  ros::Duration waitDuration_;

  Eigen::Vector3f hitDirection_[NB_ROBOTS];
  Eigen::Vector3f hitTarget_[NB_ROBOTS];
  Eigen::Vector3f refVelocity_[NB_ROBOTS];
  Eigen::Vector4f refQuat_[NB_ROBOTS];
  Eigen::Vector4f returnQuat_[NB_ROBOTS];
  Eigen::Vector3f returnPos_[NB_ROBOTS];
  float hittingFlux_[NB_ROBOTS];
  float objectMass_;
  int objectNumber_;
  std::vector<float> hittingFluxArr_;
  float objectSafetyDistance_;
  geometry_msgs::Pose refVelQuat_;

  std::string pubVelQuatTopic_[NB_ROBOTS];
  std::string pubPosQuatTopic_[NB_ROBOTS];
  std::string iiwaInertiaTopic_[NB_ROBOTS];
  std::string objectPositionTopic_;
  std::string objectPositionTopicReal_[NB_ROBOTS];
  std::string targetPositionTopicReal_[NB_ROBOTS];
  std::string iiwaPositionTopicReal_[NB_ROBOTS];
  std::string iiwaVelocityTopicReal_[NB_ROBOTS];
  std::string iiwaBasePositionTopicSim_;
  std::string pubFSMTopic_;

  ros::Rate rate_;
  ros::NodeHandle nh_;
  ros::Publisher pubVelQuat_[NB_ROBOTS];
  ros::Publisher pubPosQuat_[NB_ROBOTS];
  ros::Publisher pubFSM_;
  ros::Subscriber objectPosition_[NB_ROBOTS];
  ros::Subscriber targetPosition_[NB_ROBOTS];
  ros::Subscriber iiwaInertia_[NB_ROBOTS];
  ros::Subscriber iiwaPositionReal_[NB_ROBOTS];
  ros::Subscriber iiwaVelocityReal_[NB_ROBOTS];
  ros::Subscriber objectPositionSim_;
  ros::Subscriber iiwaBasePositionSim_;

  geometry_msgs::Pose boxPose_;
  geometry_msgs::Pose iiwaPose_[NB_ROBOTS];
  geometry_msgs::Twist iiwaVel_[NB_ROBOTS];

  Eigen::Vector3f objectPositionFromSource_;
  Eigen::Vector4f objectOrientationFromSource_;
  Eigen::Vector3f objectPositionForIiwa_[NB_ROBOTS];
  Eigen::Vector4f objectOrientationForIiwa_[NB_ROBOTS];
  Eigen::Matrix3f rotationMat_;
  Eigen::Vector3f iiwaPositionFromSource_[NB_ROBOTS];
  Eigen::Vector4f iiwaOrientationFromSource_[NB_ROBOTS];
  Eigen::Vector3f iiwaVelocityFromSource_[NB_ROBOTS];
  Eigen::Vector3f iiwaBasePositionFromSource_[NB_ROBOTS];
  Eigen::Matrix3f iiwaTaskInertiaPosInv_[NB_ROBOTS];
  Eigen::Vector3f objectOffset_[NB_ROBOTS];
  Eigen::Vector3f placementOffset_[NB_ROBOTS];

  Robot next_hit_;
  Eigen::Vector3f previousObjectPositionFromSource_;
  Eigen::Vector3f returnPosInitial_[NB_ROBOTS];

  std::unique_ptr<hitting_DS> generateHitting7_ =
      std::make_unique<hitting_DS>(iiwaPositionFromSource_[IIWA_7], objectPositionFromSource_);
  std::unique_ptr<hitting_DS> generateHitting14_ =
      std::make_unique<hitting_DS>(iiwaPositionFromSource_[IIWA_14], objectPositionFromSource_);

public:

  explicit AirHockey(ros::NodeHandle& nh, float frequency) : nh_(nh), rate_(frequency){};

  bool init();

  void run();
  void updateCurrentEEPosition(Eigen::Vector3f new_position[]);
  void publishVelQuat(Eigen::Vector3f DS_vel[], Eigen::Vector4f DS_quat[], Robot robot_name);
  void publishPosQuat(Eigen::Vector3f pos[], Eigen::Vector4f quat[], Robot robot_name);
  void publishFSM(FSMState current_state);


  int getIndex(std::vector<std::string> v, std::string value);
  void updateDSAttractor();
  void iiwaInertiaCallback(const geometry_msgs::Inertia::ConstPtr& msg, int k);
  void iiwaPositionCallbackGazebo(const gazebo_msgs::LinkStates& linkStates);
  void iiwaBasePositionCallbackGazebo(const gazebo_msgs::LinkStates& linkStates);
  void objectPositionCallbackGazebo(const gazebo_msgs::ModelStates& modelStates);
  
  void iiwaJointStateCallbackReal(const sensor_msgs::JointState::ConstPtr& msg, int k);
  void iiwaPoseCallbackReal(const geometry_msgs::Pose::ConstPtr& msg, int k);
  void iiwaVelocityCallbackReal(const geometry_msgs::Twist::ConstPtr& msg, int k);
  void objectPositionCallbackReal(const geometry_msgs::PoseStamped::ConstPtr& msg, int k);
  void targetPositionCallbackReal(const geometry_msgs::PoseStamped::ConstPtr& msg, int k);

  void updateIsObjectMoving();
  bool updateReturnPosition();
  void setReturnPositionToInitial();
  void checkObjectIsSafeToHit();
  void setObjectMass();

  void getDesiredFluxes(std::string filename);

  FSMState updateKeyboardControl(FSMState statesvar);
  void updateisPaused();

  FSMState updateFSMAutomatic(FSMState statesvar);
  FSMState preHitPlacement(FSMState statesvar);
  void set_predicted_flux();

};
