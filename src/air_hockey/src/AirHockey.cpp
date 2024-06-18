#include "AirHockey.hpp"

bool AirHockey::init() {

  // Check if sim or real
  if (!nh_.getParam("simulation_referential",isSim_)) { ROS_ERROR("Param simulation_referential not found"); }
  // Check if automatic
  if (!nh_.getParam("automatic",isAuto_)) { ROS_ERROR("Param automatic not found"); }
  // Check if aiming
  if (!nh_.getParam("aiming",isAiming_)) { ROS_ERROR("Param aiming not found"); }
  // Check if using fixed flux
  if (!nh_.getParam("fixed_flux",isFluxFixed_)) { ROS_ERROR("Param automatic not found"); }
  // Check safetz distance
  if (!nh_.getParam("safety_distance", objectSafetyDistance_)) { ROS_ERROR("Param safety distance not found"); }
  // Check tiem to wait between hits
  if (!nh_.getParam("time_to_wait",timeToWait_)) { ROS_ERROR("Param time to wait not found"); }
  waitDuration_= ros::Duration(timeToWait_);

  // Set object mass for hitting DS 
  if (!nh_.getParam("object_mass", objectMass_)) {ROS_ERROR("Param object_mass not found");}
  generateHitting7_->set_mass(objectMass_);
  generateHitting14_->set_mass(objectMass_);

  // Get topics names
  if (!nh_.getParam("recorder_topic", pubFSMTopic_)) {ROS_ERROR("Topic /recorder/robot_states not found");}

  if (!nh_.getParam("/passive_control/vel_quat_7", pubVelQuatTopic_[IIWA_7])) {ROS_ERROR("Topic /passive_control iiwa 7 not found");}
  if (!nh_.getParam("/passive_control/vel_quat_14", pubVelQuatTopic_[IIWA_14])) {ROS_ERROR("Topic /passive_control iiwa 14 not found");}

  if (!nh_.getParam("/passive_control/pos_quat_7", pubPosQuatTopic_[IIWA_7])) {ROS_ERROR("Topic /passive_control iiwa 7 not found");}
  if (!nh_.getParam("/passive_control/pos_quat_14", pubPosQuatTopic_[IIWA_14])) {ROS_ERROR("Topic /passive_control iiwa 14 not found");}

  if (!nh_.getParam("/iiwa/inertia/taskPos_7", iiwaInertiaTopic_[IIWA_7])) {ROS_ERROR("Topic /iiwa/inertia/taskPos not found");}
  if (!nh_.getParam("/iiwa/inertia/taskPos_14", iiwaInertiaTopic_[IIWA_14])) {ROS_ERROR("Topic /iiwa/inertia/taskPos not found");}

  if (!nh_.getParam("/iiwa/info_7/pose", iiwaPositionTopicReal_[IIWA_7])) {ROS_ERROR("Topic /iiwa1/ee_info/pose not found");}
  if (!nh_.getParam("/iiwa/info_14/pose", iiwaPositionTopicReal_[IIWA_14])) {ROS_ERROR("Topic /iiwa2/ee_info/pose not found");}
  if (!nh_.getParam("/iiwa/info_7/vel", iiwaVelocityTopicReal_[IIWA_7])) {ROS_ERROR("Topic /iiwa1/ee_info/vel not found");}
  if (!nh_.getParam("/iiwa/info_14/vel", iiwaVelocityTopicReal_[IIWA_14])) {ROS_ERROR("Topic /iiwa2/ee_info/vel not found");}

  if(isSim_){
    if (!nh_.getParam("/gazebo/link_states", iiwaBasePositionTopicSim_)) {ROS_ERROR("Topic /gazebo/link_states not found");}
    if (!nh_.getParam("/gazebo/model_states", objectPositionTopic_)) {ROS_ERROR("Topic /gazebo/model_states not found");}
  }

  else if (!isSim_){
    if (!nh_.getParam("/optitrack/object_from_base_1/pose", objectPositionTopicReal_[IIWA_7])) {ROS_ERROR("Topic /optitrack/object_from_base_1/pose not found");}
    if (!nh_.getParam("/optitrack/object_from_base_2/pose", objectPositionTopicReal_[IIWA_14])) {ROS_ERROR("Topic /optitrack/object_from_base_2/pose not found");}
  }
  
  // Init publishers
  pubVelQuat_[IIWA_7] = nh_.advertise<geometry_msgs::Pose>(pubVelQuatTopic_[IIWA_7], 1);
  pubVelQuat_[IIWA_14] = nh_.advertise<geometry_msgs::Pose>(pubVelQuatTopic_[IIWA_14], 1);
  pubPosQuat_[IIWA_7] = nh_.advertise<geometry_msgs::Pose>(pubPosQuatTopic_[IIWA_7], 1);
  pubPosQuat_[IIWA_14] = nh_.advertise<geometry_msgs::Pose>(pubPosQuatTopic_[IIWA_14], 1);
  pubFSM_ = nh_.advertise<air_hockey::FSM_state>(pubFSMTopic_, 1);

  // Init subscribers
  if(isSim_){
    objectPositionSim_ = nh_.subscribe(objectPositionTopic_,
                                  1,
                                  &AirHockey::objectPositionCallbackGazebo,
                                  this,
                                  ros::TransportHints().reliable().tcpNoDelay());
    
    iiwaBasePositionSim_ = nh_.subscribe(iiwaBasePositionTopicSim_,
                                1,
                                &AirHockey::iiwaBasePositionCallbackGazebo,
                                this,
                                ros::TransportHints().reliable().tcpNoDelay());
  }
  else if (!isSim_){
    objectPosition_[IIWA_7] = 
        nh_.subscribe<geometry_msgs::PoseStamped>(objectPositionTopicReal_[IIWA_7],
                                                  1,
                                                  boost::bind(&AirHockey::objectPositionCallbackReal, this, _1, IIWA_7),
                                                  ros::VoidPtr(),
                                                  ros::TransportHints().reliable().tcpNoDelay());
  
    objectPosition_[IIWA_14] = 
        nh_.subscribe<geometry_msgs::PoseStamped>(objectPositionTopicReal_[IIWA_14],
                                                  1,
                                                  boost::bind(&AirHockey::objectPositionCallbackReal, this, _1, IIWA_14),
                                                  ros::VoidPtr(),
                                                  ros::TransportHints().reliable().tcpNoDelay());
  }

  iiwaPositionReal_[IIWA_7] = 
        nh_.subscribe<geometry_msgs::Pose>(iiwaPositionTopicReal_[IIWA_7],
                                            1,
                                            boost::bind(&AirHockey::iiwaPoseCallbackReal, this, _1, IIWA_7),
                                            ros::VoidPtr(),
                                            ros::TransportHints().reliable().tcpNoDelay());    

  iiwaPositionReal_[IIWA_14] = 
      nh_.subscribe<geometry_msgs::Pose>(iiwaPositionTopicReal_[IIWA_14],
                                          1,
                                          boost::bind(&AirHockey::iiwaPoseCallbackReal, this, _1, IIWA_14),
                                          ros::VoidPtr(),
                                          ros::TransportHints().reliable().tcpNoDelay());

  iiwaVelocityReal_[IIWA_7] = 
      nh_.subscribe<geometry_msgs::Twist>(iiwaVelocityTopicReal_[IIWA_7],
                                          1,
                                          boost::bind(&AirHockey::iiwaVelocityCallbackReal, this, _1, IIWA_7),
                                          ros::VoidPtr(),
                                          ros::TransportHints().reliable().tcpNoDelay());    

  iiwaVelocityReal_[IIWA_14] = 
      nh_.subscribe<geometry_msgs::Twist>(iiwaVelocityTopicReal_[IIWA_14],
                                          1,
                                          boost::bind(&AirHockey::iiwaVelocityCallbackReal, this, _1, IIWA_14),
                                          ros::VoidPtr(),
                                          ros::TransportHints().reliable().tcpNoDelay());

  iiwaInertia_[IIWA_7] =
      nh_.subscribe<geometry_msgs::Inertia>(iiwaInertiaTopic_[IIWA_7],
                                            1,
                                            boost::bind(&AirHockey::iiwaInertiaCallback, this, _1, IIWA_7),
                                            ros::VoidPtr(),
                                            ros::TransportHints().reliable().tcpNoDelay());
  iiwaInertia_[IIWA_14] =
      nh_.subscribe<geometry_msgs::Inertia>(iiwaInertiaTopic_[IIWA_14],
                                            1,
                                            boost::bind(&AirHockey::iiwaInertiaCallback, this, _1, IIWA_14),
                                            ros::VoidPtr(),
                                            ros::TransportHints().reliable().tcpNoDelay());


  // get object offset values
  if (!nh_.getParam("iiwa7/object_offset/x", objectOffset_[IIWA_7][0])) { ROS_ERROR("Topic iiwa7/object_offset/x not found"); }
  if (!nh_.getParam("iiwa7/object_offset/y", objectOffset_[IIWA_7][1])) { ROS_ERROR("Topic iiwa7/object_offset/y not found"); }
  if (!nh_.getParam("iiwa7/object_offset/z", objectOffset_[IIWA_7][2])) { ROS_ERROR("Topic iiwa7/object_offset/z not found"); }
  if (!nh_.getParam("iiwa14/object_offset/x", objectOffset_[IIWA_14][0])) { ROS_ERROR("Topic iiwa14/object_offset/x not found"); }
  if (!nh_.getParam("iiwa14/object_offset/y", objectOffset_[IIWA_14][1])) { ROS_ERROR("Topic iiwa14/object_offset/y not found"); }
  if (!nh_.getParam("iiwa14/object_offset/z", objectOffset_[IIWA_14][2])) { ROS_ERROR("Topic iiwa14/object_offset/x not found"); }

  generateHitting7_->set_current_position(iiwaPositionFromSource_[IIWA_7]);
  generateHitting14_->set_current_position(iiwaPositionFromSource_[IIWA_14]);
  this->updateDSAttractor(); //update attractor position

  // Get hitting parameters
  std::vector<double> dquat1;
  std::vector<double> dquat2;
  if (!nh_.getParam("/iiwa1/target/iiwa1/quat", dquat1)){
    if (!nh_.getParam("/iiwa2/target/iiwa1/quat", dquat1)){
      ROS_ERROR("Param /iiwa1/target/iiwa1/quat and /iiwa2/target/iiwa1/quat not found"); }}
  for (size_t i = 0; i < refQuat_[IIWA_7].size(); i++)
    refQuat_[IIWA_7](i) = dquat1[i]; 
  if (!nh_.getParam("/iiwa1/target/iiwa2/quat", dquat2)){ 
    if (!nh_.getParam("/iiwa2/target/iiwa2/quat", dquat2)){ 
      ROS_ERROR("Param /iiwa1/target/iiwa2/quat and /iiwa2/target/iiwa2/quat not found"); }}
  for (size_t i = 0; i < refQuat_[IIWA_14].size(); i++)
    refQuat_[IIWA_14](i) = dquat2[i]; 

  if (!nh_.getParam("iiwa7/return_position/x", returnPosInitial_[IIWA_7][0])) { ROS_ERROR("Param ref_quat/x not found"); }
  if (!nh_.getParam("iiwa14/return_position/x", returnPosInitial_[IIWA_14][0])) { ROS_ERROR("Param return_position/x not found"); }
  if (!nh_.getParam("iiwa7/return_position/y", returnPosInitial_[IIWA_7][1])) { ROS_ERROR("Param return_position/y not found"); }
  if (!nh_.getParam("iiwa14/return_position/y", returnPosInitial_[IIWA_14][1])) { ROS_ERROR("Param return_position/y not found"); }
  if (!nh_.getParam("iiwa7/return_position/z", returnPosInitial_[IIWA_7][2])) { ROS_ERROR("Param return_position/z not found"); }
  if (!nh_.getParam("iiwa14/return_position/z", returnPosInitial_[IIWA_14][2])) { ROS_ERROR("Param return_position/z not found"); }
  if (!nh_.getParam("iiwa7/hit_direction/x", hitDirection_[IIWA_7][0])) {ROS_ERROR("Param hit_direction/x not found");}
  if (!nh_.getParam("iiwa14/hit_direction/x", hitDirection_[IIWA_14][0])) {ROS_ERROR("Param hit_direction/x not found");}
  if (!nh_.getParam("iiwa7/hit_direction/y", hitDirection_[IIWA_7][1])) {ROS_ERROR("Param hit_direction/y not found");}
  if (!nh_.getParam("iiwa14/hit_direction/y", hitDirection_[IIWA_14][1])) {ROS_ERROR("Param hit_direction/y not found");}
  if (!nh_.getParam("iiwa7/hit_direction/z", hitDirection_[IIWA_7][2])) {ROS_ERROR("Param hit_direction/z not found");}
  if (!nh_.getParam("iiwa14/hit_direction/z", hitDirection_[IIWA_14][2])) {ROS_ERROR("Param hit_direction/z not found");}
  if (!nh_.getParam("iiwa7/hitting_flux", hittingFlux_[IIWA_7])) {ROS_ERROR("Param iiwa7/hitting_flux not found");}
  if (!nh_.getParam("iiwa14/hitting_flux", hittingFlux_[IIWA_14])) {ROS_ERROR("Param iiwa14/hitting_flux not found");}

  if (!nh_.getParam("iiwa7/placement_offset/x", placementOffset_[IIWA_7][0])) { ROS_ERROR("Topic iiwa7/placement_offset/x not found"); }
  if (!nh_.getParam("iiwa7/placement_offset/y", placementOffset_[IIWA_7][1])) { ROS_ERROR("Topic iiwa7/placement_offset/y not found"); }
  if (!nh_.getParam("iiwa7/placement_offset/z", placementOffset_[IIWA_7][2])) { ROS_ERROR("Topic iiwa7/placement_offset/z not found"); }
  if (!nh_.getParam("iiwa14/placement_offset/x", placementOffset_[IIWA_14][0])) { ROS_ERROR("Topic iiwa14/placement_offset/x not found"); }
  if (!nh_.getParam("iiwa14/placement_offset/y", placementOffset_[IIWA_14][1])) { ROS_ERROR("Topic iiwa14/placement_offset/y not found"); }
  if (!nh_.getParam("iiwa14/placement_offset/z", placementOffset_[IIWA_14][2])) { ROS_ERROR("Topic iiwa14/placement_offset/x not found"); }

  if (!nh_.getParam("iiwa7/hit_target/x", hitTarget_[IIWA_7][0])) { ROS_ERROR("Topic iiwa7/hit_target/x not found"); }
  if (!nh_.getParam("iiwa7/hit_target/y", hitTarget_[IIWA_7][1])) { ROS_ERROR("Topic iiwa7/hit_target/y not found"); }
  if (!nh_.getParam("iiwa7/hit_target/z", hitTarget_[IIWA_7][2])) { ROS_ERROR("Topic iiwa7/hit_target/z not found"); }
  if (!nh_.getParam("iiwa14/hit_target/x", hitTarget_[IIWA_14][0])) { ROS_ERROR("Topic iiwa14/hit_target/x not found"); }
  if (!nh_.getParam("iiwa14/hit_target/y", hitTarget_[IIWA_14][1])) { ROS_ERROR("Topic iiwa14/hit_target/y not found"); }
  if (!nh_.getParam("iiwa14/hit_target/z", hitTarget_[IIWA_14][2])) { ROS_ERROR("Topic iiwa14/hit_target/x not found"); }

  generateHitting7_->set_des_direction(hitDirection_[IIWA_7]);
  generateHitting14_->set_des_direction(hitDirection_[IIWA_14]);

  // Initialize PAUSE state
  isPaused_ = true;
  if(isAuto_){next_hit_ = IIWA_7;}
  else if(!isAuto_){next_hit_=NONE;}

  // Set return pos to initial
  setReturnPositionToInitial();

  // get desired hitting fluxes from file in config folder
  if(!isFluxFixed_ && !isAiming_){
    std::string flux_fn;
    if (!nh_.getParam("desired_fluxes_filename",flux_fn)) { ROS_ERROR("Param desired_fluxes_filename not found"); }

    flux_fn = "/home/ros/ros_ws/src/air_hockey/desired_hitting_fluxes/" + flux_fn;
    this->getDesiredFluxes(flux_fn);

    hittingFlux_[IIWA_7] = hittingFluxArr_[0];
    hittingFlux_[IIWA_14] = hittingFluxArr_[0];
  }

  // Get starting flux 
  if(isAiming_){
    ROS_INFO("USING GMR PREDICTION FOR FLUX VALUES");
    set_predicted_flux();
  }

  return true;
}

// CALLBACKS
void AirHockey::objectPositionCallbackGazebo(const gazebo_msgs::ModelStates& modelStates) {
  int boxIndex = getIndex(modelStates.name, "box_model");
  boxPose_ = modelStates.pose[boxIndex];
  objectPositionFromSource_ << boxPose_.position.x, boxPose_.position.y, boxPose_.position.z;
  objectOrientationFromSource_ << boxPose_.orientation.x, boxPose_.orientation.y, boxPose_.orientation.z,
      boxPose_.orientation.w;
}

void AirHockey::iiwaPositionCallbackGazebo(const gazebo_msgs::LinkStates& linkStates) {
  int indexIiwa1 = getIndex(linkStates.name, "iiwa1::iiwa1_link_7");// End effector is the 7th link in KUKA IIWA
  int indexIiwa2 = getIndex(linkStates.name, "iiwa2::iiwa2_link_7");// End effector is the 7th link in KUKA IIWA

  iiwaPose_[IIWA_7] = linkStates.pose[indexIiwa1];
  iiwaPositionFromSource_[IIWA_7] << iiwaPose_[IIWA_7].position.x, iiwaPose_[IIWA_7].position.y,
      iiwaPose_[IIWA_7].position.z;
  iiwaOrientationFromSource_[IIWA_7] << iiwaPose_[IIWA_7].orientation.x, iiwaPose_[IIWA_7].orientation.y, 
      iiwaPose_[IIWA_7].orientation.z, iiwaPose_[IIWA_7].orientation.w;    

  iiwaVel_[IIWA_7] = linkStates.twist[indexIiwa1];
  iiwaVelocityFromSource_[IIWA_7] << iiwaVel_[IIWA_7].linear.x, iiwaVel_[IIWA_7].linear.y,
      iiwaVel_[IIWA_7].linear.z;

  iiwaPose_[IIWA_14] = linkStates.pose[indexIiwa2];
  iiwaPositionFromSource_[IIWA_14] << iiwaPose_[IIWA_14].position.x, iiwaPose_[IIWA_14].position.y,
      iiwaPose_[IIWA_14].position.z;
  iiwaOrientationFromSource_[IIWA_14] << iiwaPose_[IIWA_14].orientation.x, iiwaPose_[IIWA_14].orientation.y, 
      iiwaPose_[IIWA_14].orientation.z, iiwaPose_[IIWA_14].orientation.w; 

  iiwaVel_[IIWA_14] = linkStates.twist[indexIiwa2];
  iiwaVelocityFromSource_[IIWA_14] << iiwaVel_[IIWA_14].linear.x, iiwaVel_[IIWA_14].linear.y,
      iiwaVel_[IIWA_14].linear.z;
}

void AirHockey::iiwaBasePositionCallbackGazebo(const gazebo_msgs::LinkStates& linkStates) {
  int indexIiwa1 = getIndex(linkStates.name, "iiwa1::iiwa1_link_0");// End effector is the 7th link in KUKA IIWA
  int indexIiwa2 = getIndex(linkStates.name, "iiwa2::iiwa2_link_0");// End effector is the 7th link in KUKA IIWA

  iiwaBasePositionFromSource_[IIWA_7] << linkStates.pose[indexIiwa1].position.x, linkStates.pose[indexIiwa1].position.y,
      linkStates.pose[indexIiwa1].position.z;
  
  iiwaBasePositionFromSource_[IIWA_14] << linkStates.pose[indexIiwa2].position.x, linkStates.pose[indexIiwa2].position.y,
      linkStates.pose[indexIiwa2].position.z;
 
}

void AirHockey::iiwaInertiaCallback(const geometry_msgs::Inertia::ConstPtr& msg, int k) {
  iiwaTaskInertiaPosInv_[k](0, 0) = msg->ixx;
  iiwaTaskInertiaPosInv_[k](2, 2) = msg->izz;
  iiwaTaskInertiaPosInv_[k](1, 1) = msg->iyy;
  iiwaTaskInertiaPosInv_[k](0, 1) = msg->ixy;
  iiwaTaskInertiaPosInv_[k](1, 0) = msg->ixy;
  iiwaTaskInertiaPosInv_[k](0, 2) = msg->ixz;
  iiwaTaskInertiaPosInv_[k](2, 0) = msg->ixz;
  iiwaTaskInertiaPosInv_[k](1, 2) = msg->iyz;
  iiwaTaskInertiaPosInv_[k](2, 1) = msg->iyz;
}

void AirHockey::iiwaPoseCallbackReal(const geometry_msgs::Pose::ConstPtr& msg, int k){
  iiwaPositionFromSource_[k]  << msg->position.x, msg->position.y, msg->position.z;
  iiwaOrientationFromSource_[k] << msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w;
  iiwaPose_[k] = *msg;
}

void AirHockey::iiwaVelocityCallbackReal(const geometry_msgs::Twist::ConstPtr& msg, int k){
  iiwaVelocityFromSource_[k]  << msg->linear.x, msg->linear.y, msg->linear.z;
  iiwaVel_[k] = *msg;
}

void AirHockey::objectPositionCallbackReal(const geometry_msgs::PoseStamped::ConstPtr& msg, int k){
  objectPositionForIiwa_[k] << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  objectOrientationForIiwa_[k] << msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z,
      msg->pose.orientation.w;
}

// UPDATES AND CALCULATIONS

void AirHockey::updateDSAttractor() {

  if(isSim_){ // avoid using offset -> TO CHANGE LATER 
    objectPositionForIiwa_[IIWA_7] = objectPositionFromSource_ - iiwaBasePositionFromSource_[IIWA_7];
    objectPositionForIiwa_[IIWA_14] = objectPositionFromSource_ - iiwaBasePositionFromSource_[IIWA_14];

    generateHitting7_->set_DS_attractor(objectPositionForIiwa_[IIWA_7]);
    generateHitting14_->set_DS_attractor(objectPositionForIiwa_[IIWA_14]);
  }
  else if(!isSim_){
    generateHitting7_->set_DS_attractor(objectPositionForIiwa_[IIWA_7]+ objectOffset_[IIWA_7]);
    generateHitting14_->set_DS_attractor(objectPositionForIiwa_[IIWA_14]+ objectOffset_[IIWA_14]);
  }
}

int AirHockey::getIndex(std::vector<std::string> v, std::string value) {
  for (int i = 0; i < v.size(); i++) {
    if (v[i].compare(value) == 0) return i;
  }
  return -1;
}

void AirHockey::checkObjectIsSafeToHit() {
  
  if(isSim_){
    if(objectPositionFromSource_.norm() > 2*objectSafetyDistance_){
      ROS_INFO("Object is too far away, PAUSING system!");
      isPaused_ = true;
    }
  }
  else if(!isSim_){
    if((objectPositionForIiwa_[IIWA_7].norm() > objectSafetyDistance_) && (objectPositionForIiwa_[IIWA_14].norm() > objectSafetyDistance_)){
      ROS_INFO("Object is too far away, PAUSING system!");
      isPaused_ = true;
    }  
  }
}

void AirHockey::updateIsObjectMoving(){

  // Thershold for detecting movement
  float object_stopped_threshold = 2*1e-4;

  float object_speed = (objectPositionForIiwa_[IIWA_7]-previousObjectPositionFromSource_).norm();

  if(object_speed > object_stopped_threshold){
    isObjectMoving_ = true;
  }
  else{
    isObjectMoving_ = false;
  }

  // Update previous object pos
  previousObjectPositionFromSource_ = objectPositionForIiwa_[IIWA_7];
}

bool AirHockey::updateReturnPosition(){

  if(next_hit_ == IIWA_7){
    auto temp_pos = generateHitting7_->get_DS_attractor() + placementOffset_[IIWA_7];

    // HARD CODED LIMITS to avoid oscillations in edge cases
    if((temp_pos[0] < 0.65 || temp_pos[1] > 0.0) && temp_pos.norm() < objectSafetyDistance_){
      returnPos_[IIWA_7] = temp_pos;
      return true;
    }
    else{ // if(temp_pos[0] >= 0.65 && temp_pos[1] <= 0.0){
      ROS_WARN("Object too far for pre-hit placement, REPLACE OBJECT!");
      return false;
    } 
  }
  else if(next_hit_ == IIWA_14){
    auto temp_pos = generateHitting14_->get_DS_attractor() + placementOffset_[IIWA_14];
    if((temp_pos[0] < 0.65 || temp_pos[1] < 0.0) && temp_pos.norm() < objectSafetyDistance_){
      returnPos_[IIWA_14] = temp_pos;
      return true;
    }
    else{ //} if(temp_pos[0] >= 0.65 && temp_pos[1] >= 0.0 ){
      ROS_WARN("Object too far for pre-hit placement, REPLACE OBJECT!");
      return false;
    } 
  }
  
  return false;
}

void AirHockey::setReturnPositionToInitial(){
  returnPos_[IIWA_7] = returnPosInitial_[IIWA_7];
  returnPos_[IIWA_14] = returnPosInitial_[IIWA_14];
}

void AirHockey::updateCurrentEEPosition(Eigen::Vector3f new_position[]) {
  generateHitting7_->set_current_position(new_position[IIWA_7]);
  generateHitting14_->set_current_position(new_position[IIWA_14]);
}

// GET DESIRED FLUX FILE
void AirHockey::getDesiredFluxes(std::string filename){
    
    std::ifstream inputFile(filename);

    // Check if the file is opened successfully
    if (!inputFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    // Read file line by line
    std::string line;
    while (std::getline(inputFile, line)) {
        // Create a stringstream to parse the line
        std::istringstream iss(line);

        // Variables to store float values
        float value;
        char delimiter; // Assuming values are separated by some delimiter like space or comma

        // Read each value from the line
        while (iss >> value) {
            hittingFluxArr_.push_back(value);
            // Check for missing values
            if (!(iss >> delimiter)) {
                break; // Missing value detected, break out of the loop
            }
        }
    }

    inputFile.close();
}

// PUBLISHERS
void AirHockey::publishVelQuat(Eigen::Vector3f DS_vel[], Eigen::Vector4f DS_quat[], Robot robot_name) {
  geometry_msgs::Pose ref_vel_publish;
  ref_vel_publish.position.x = DS_vel[robot_name](0);
  ref_vel_publish.position.y = DS_vel[robot_name](1);
  ref_vel_publish.position.z = DS_vel[robot_name](2);
  ref_vel_publish.orientation.w = DS_quat[robot_name](0);
  ref_vel_publish.orientation.x = DS_quat[robot_name](1);
  ref_vel_publish.orientation.y = DS_quat[robot_name](2);
  ref_vel_publish.orientation.z = DS_quat[robot_name](3);
  pubVelQuat_[robot_name].publish(ref_vel_publish);
}

// used only for return position, publishes return position for one robot
void AirHockey::publishPosQuat(Eigen::Vector3f pos[], Eigen::Vector4f quat[], Robot robot_name) {
  geometry_msgs::Pose ref_pose_publish;
  ref_pose_publish.position.x = pos[robot_name](0);
  ref_pose_publish.position.y = pos[robot_name](1);
  ref_pose_publish.position.z = pos[robot_name](2);
  ref_pose_publish.orientation.w = quat[robot_name](0);
  ref_pose_publish.orientation.x = quat[robot_name](1);
  ref_pose_publish.orientation.y = quat[robot_name](2);
  ref_pose_publish.orientation.z = quat[robot_name](3);
  pubPosQuat_[robot_name].publish(ref_pose_publish);
}

void AirHockey::publishFSM(FSMState current_state){
  air_hockey::FSM_state msg;
  msg.mode_iiwa7 = static_cast<uint8_t>(current_state.mode_iiwa7);
  msg.mode_iiwa14 = static_cast<uint8_t>(current_state.mode_iiwa14);
  msg.isHit = current_state.isHit;
  msg.hit_time = current_state.hit_time;
  msg.des_flux = current_state.des_flux;
  msg.des_pos_x = current_state.des_pos(0);
  msg.des_pos_y = current_state.des_pos(1);
  msg.des_pos_z = current_state.des_pos(2);

  pubFSM_.publish(msg);
}

// KEYBOARD INTERACTIONS
AirHockey::FSMState AirHockey::updateKeyboardControl(FSMState current_state ) {

  nonBlock(1);

  if (khBit() != 0) {
    char keyboardCommand = fgetc(stdin);
    fflush(stdin);

    switch (keyboardCommand) {
      case 'q': {
        next_hit_ = IIWA_7;
        // current_state.mode_iiwa7 = HIT;
        std::cout << "q is pressed \n " << std::endl;
        
      } break;
      case 'p': {
        next_hit_ = IIWA_14;
        // current_state.mode_iiwa14 = HIT;
        std::cout << "p is pressed \n" << std::endl;
      } break;
      case 'r': {
        current_state.mode_iiwa7 = REST;
        current_state.mode_iiwa14 = REST;
        next_hit_ = NONE;
        setReturnPositionToInitial();
        std::cout << "r is pressed \n" << std::endl;
      } break;
      case 'h': { // toggle isHit 
        if(current_state.isHit){ current_state.isHit = 0;}
        else if(!current_state.isHit){ current_state.isHit= 1;}
      } break;

    }
  }
  nonBlock(0);

  return current_state;
}

void AirHockey::updateisPaused() {

  nonBlock(1);

  if (khBit() != 0) {
    char keyboardCommand = fgetc(stdin);
    fflush(stdin);

    if (keyboardCommand == ' ') {
        std::cout << "You pressed the space bar." << std::endl;

        // Switch Pause state
        if(isPaused_){isPaused_= false;}
        else if (!isPaused_){isPaused_ = true;}

    } else {
        printf("You pressed a different key.\n");
    }
  }
  nonBlock(0);
}

// AUTONOMOUS FSM
AirHockey::FSMState AirHockey::updateFSMAutomatic(FSMState current_state ) {

  // if PAUSED -> both robots to REST and wait for further input (return immediately)
  if(isPaused_)
  {
    setReturnPositionToInitial();
    current_state.mode_iiwa7 = REST;
    current_state.mode_iiwa14 = REST;
    return current_state;
  }

  // if AUTO -> check robots are ready to hit then set to hit
  float pos_threshold = 4*1e-2;
  float vel_threshold = 1*1e-3;

  // Only set to HIt if both robots are at rest!
  if(current_state.mode_iiwa7 == REST && current_state.mode_iiwa14 == REST){
    
    // Stay in REST for desired duration
    auto time_in_rest = ros::Time::now() - current_state.hit_time;
    if(time_in_rest < waitDuration_){ 
      return current_state;
    }

    // is object stopped?
    else if(!isObjectMoving_){
      // Go to pre hit placement
      current_state = preHitPlacement(current_state);
    }
  }

  return current_state;
}

AirHockey::FSMState AirHockey::preHitPlacement(FSMState current_state ) {

  float pos_threshold_7 = 2*1e-2;
  float pos_threshold_14 = 4*1e-2;
  float vel_threshold = 1*1e-3;
  
  // update return position, if not possible, do not change state 
  if(!updateReturnPosition())
    return current_state;

  // Get norms
  float norm_iiwa7 = (iiwaPositionFromSource_[IIWA_7]-returnPos_[IIWA_7]).norm();
  float norm_iiwa14 = (iiwaPositionFromSource_[IIWA_14]-returnPos_[IIWA_14]).norm();

  if(!isAuto_){ // Then set to HIT depending on norm of individual robot 
      if(next_hit_ == IIWA_7 && norm_iiwa7 < pos_threshold_7 && iiwaVelocityFromSource_[IIWA_7].norm() < vel_threshold ){
      current_state.mode_iiwa7 = HIT;
      next_hit_ = NONE;
      setReturnPositionToInitial();
    }
    if(next_hit_ == IIWA_14 && norm_iiwa14 < pos_threshold_14 && iiwaVelocityFromSource_[IIWA_14].norm() < vel_threshold){
      current_state.mode_iiwa14 = HIT;
      next_hit_ = NONE;
      setReturnPositionToInitial();
    }
  }
  if(isAuto_){ // Then set to HIT depending on norm of both robots
    if(norm_iiwa7 < pos_threshold_7 && norm_iiwa14 < pos_threshold_14 && 
            iiwaVelocityFromSource_[IIWA_7].norm() < vel_threshold && 
            iiwaVelocityFromSource_[IIWA_14].norm() < vel_threshold){

        // Then set to HIT and update next_hit
        if(next_hit_ == IIWA_7){
          current_state.mode_iiwa7 = HIT;
          next_hit_ = IIWA_14;
        }
        else if(next_hit_ == IIWA_14){
          current_state.mode_iiwa14 = HIT;
          next_hit_ = IIWA_7;
        }
        // Start by going back to usual pos before adapting to object
        setReturnPositionToInitial();
      }
  }
  
  return current_state;
}

void AirHockey::set_predicted_flux(){

  ros::ServiceClient client = nh_.serviceClient<air_hockey::Prediction>("prediction");
  air_hockey::Prediction srv;

  // Get distance between box and target 
  srv.request.distance_iiwa7 = (hitTarget_[IIWA_7] - objectPositionForIiwa_[IIWA_7]).norm(); 
  srv.request.distance_iiwa14 = (hitTarget_[IIWA_14] - objectPositionForIiwa_[IIWA_14]).norm();

  if (client.call(srv))
  {
    ROS_INFO("Received flux_1: %f, flux_2: %f", srv.response.flux_iiwa7, srv.response.flux_iiwa14);
    hittingFlux_[IIWA_7] = srv.response.flux_iiwa7;
    hittingFlux_[IIWA_14] = srv.response.flux_iiwa14;
  }
  else
  {
    ROS_ERROR("Failed to call service prediction");
  }
}

void AirHockey::run() {

  // Set up counters and bool variables
  int print_count = 0;
  int display_pause_count = 0;
  int hit_count = 1;
  int update_flux_once = 0;

  FSMState fsm_state;

  std::cout << "READY TO RUN " << std::endl;

  if(isAuto_){ // DISPLAY warnings
    std::cout << "RUNNING AUTOMATICALLY!! " << std::endl;
    std::cout << "Starting with iiwa7, press Space to Start/Pause system" << std::endl;
  }

  if(!isFluxFixed_){ // display desired hitting flux 
      std::cout << "Hitting Fluxes : [ ";
      for (float val :hittingFluxArr_) {
          std::cout << val << " ";
      }
      std::cout << "]" << std::endl;
  }

  while (ros::ok()) {

    // Use keyboard control if is not automatic (set in yaml file)
    if(!isAuto_) {
      fsm_state = updateKeyboardControl(fsm_state); 
      fsm_state = preHitPlacement(fsm_state);
      }
    // Otherwise update FSM automatically
    else if (isAuto_)
    {
      // function call to check keyboard, see if it is paused
      updateisPaused();

      // check if object is within range (for safety)
      checkObjectIsSafeToHit();

      // check if object is moving and update
      updateIsObjectMoving();

      // Display Pause State every second
      if(display_pause_count%600 == 0 ){
        if(isPaused_){
          std::cout << "System is PAUSED ! (Press Space to start)" << std::endl;
          if(next_hit_ == IIWA_7){std::cout << "Next hit is from IIWA 7" << std::endl;}
          else if(next_hit_ == IIWA_14){std::cout << "Next hit is from IIWA 14" << std::endl;}
          }
        if(!isPaused_){std::cout << "System is RUNNING autonomously. Watch out! (Press Space to pause)" << std::endl;}
      }
      display_pause_count +=1 ;

      // function call to update FSM auto
      fsm_state = updateFSMAutomatic(fsm_state);
    }
    
    // Publish for Recorder
    publishFSM(fsm_state);

    // DEBUG
    if(print_count%200 == 0 ){

      //Call prediciton service 
      set_predicted_flux();

      // std::cout << "iiwa7_state : " << fsm_state.mode_iiwa7 << " \n iiwa14_state : " << fsm_state.mode_iiwa14<< std::endl;
      // std::cout << "object source pos  " << objectPositionFromSource_ << std::endl;
      // std::cout << "iiwaPos_7  " << iiwaPositionFromSource_[IIWA_7]<< std::endl;
      // std::cout << "iiwaPos_14  " << iiwaPositionFromSource_[IIWA_14]<< std::endl;
      std::cout << "iiwa7 norm  " << (iiwaPositionFromSource_[IIWA_7]-returnPos_[IIWA_7]).norm()<< std::endl;
      std::cout << "iiwa14 norm " << (iiwaPositionFromSource_[IIWA_14]-returnPos_[IIWA_14]).norm()<< std::endl;
      // std::cout << "object pos by iiwaPos_7  " << objectPositionForIiwa_[IIWA_7]<< std::endl;
      // std::cout << "object pos by  iiwaPos_14  " << objectPositionForIiwa_[IIWA_14]<< std::endl;
      // std::cout << "returnPos_7  " << returnPos_[IIWA_7]<< std::endl;
      // std::cout << "returnPos_14  " << returnPos_[IIWA_14]<< std::endl; //objectPositionForIiwa_[IIWA_7];  
      std::cout << "ref Vel 7 " << refVelocity_[IIWA_7]<< std::endl;
      std::cout << " ref quat 7 " << refQuat_[IIWA_7] << std::endl;   
    }
    print_count +=1 ;

    // Update DS attractor if at rest
    if(fsm_state.mode_iiwa7 == REST && fsm_state.mode_iiwa14 == REST){
      // update only at REST so object state conditions for isHit works 
      updateDSAttractor();
    }

    // UPDATE robot state
    if(fsm_state.mode_iiwa7 == HIT){
      if(isAiming_){
        auto refVelQuat = generateHitting7_->flux_DS_with_quat(hittingFlux_[IIWA_7], hitTarget_[IIWA_7], iiwaTaskInertiaPosInv_[IIWA_7]);
        refVelocity_[IIWA_7] = refVelQuat.first;
        refQuat_[IIWA_7] = refVelQuat.second;
      }
      else{
        refVelocity_[IIWA_7] = generateHitting7_->flux_DS(hittingFlux_[IIWA_7], iiwaTaskInertiaPosInv_[IIWA_7]);
      }
  
      update_flux_once = 1; // only update after 1 hit from each robot
    }

    if(fsm_state.mode_iiwa14 == HIT){
      if(isAiming_){
        auto refVelQuat = generateHitting14_->flux_DS_with_quat(hittingFlux_[IIWA_14], hitTarget_[IIWA_14], iiwaTaskInertiaPosInv_[IIWA_14]);
        refVelocity_[IIWA_14] = refVelQuat.first;
        refQuat_[IIWA_14] = refVelQuat.second;
      }
      else if(!isAiming_){
        refVelocity_[IIWA_14] = generateHitting14_->flux_DS(hittingFlux_[IIWA_14], iiwaTaskInertiaPosInv_[IIWA_14]);
      }
    
      // update_flux_once = 1; // only update after 1 hit from each robot
    }

    if(fsm_state.mode_iiwa7 == REST || fsm_state.isHit == 1){
      refVelocity_[IIWA_7] = generateHitting7_->linear_DS(returnPos_[IIWA_7]);
      fsm_state.mode_iiwa7 = REST;
      if (fsm_state.mode_iiwa14 == REST) { // only reset if 14 is at rest, otherwise it skips next if and never send iiwa14 to rest
              fsm_state.isHit = 0;
      }
    }

    if(fsm_state.mode_iiwa14 == REST || fsm_state.isHit == 1){
      refVelocity_[IIWA_14] = generateHitting14_->linear_DS(returnPos_[IIWA_14]);
      fsm_state.mode_iiwa14 = REST;
      fsm_state.isHit = 0;
    }

    // Update isHit and FSM state for recorder
    if (!fsm_state.isHit && generateHitting7_->get_des_direction().dot(generateHitting7_->get_DS_attractor()
                                                      - generateHitting7_->get_current_position()) < 0) {
      fsm_state.isHit = 1;
      fsm_state.hit_time = ros::Time::now();
      fsm_state.des_flux = hittingFlux_[IIWA_7];
      fsm_state.des_pos = generateHitting7_->get_DS_attractor();
    }

    if (!fsm_state.isHit && generateHitting14_->get_des_direction().dot(generateHitting14_->get_DS_attractor()
                                                      - generateHitting14_->get_current_position())  < 0) {
      fsm_state.isHit = 1;
      fsm_state.hit_time = ros::Time::now();
      fsm_state.des_flux = hittingFlux_[IIWA_14];
      fsm_state.des_pos = generateHitting14_->get_DS_attractor();     
    }

    // Update hitting flux if needed
    if(!isFluxFixed_ && fsm_state.mode_iiwa7 == REST && fsm_state.mode_iiwa14 == REST && update_flux_once){
        hittingFlux_[IIWA_7] = hittingFluxArr_[hit_count];
        hittingFlux_[IIWA_14] = hittingFluxArr_[hit_count];
        
        update_flux_once = 0;
        hit_count += 1; 
        if(hit_count >= hittingFluxArr_.size()){hit_count = 0;} // loop flux file 
        
        std::cout << "Hitting Flux for next 2 hits : " << hittingFlux_[IIWA_7] << std::endl; 
    }

    // update iiwa pos in DS
    updateCurrentEEPosition(iiwaPositionFromSource_);

    // Publisher logic to use publish position when returning (avoids inertia in iiwa_toolkit)
    if(fsm_state.mode_iiwa7 == HIT){ publishVelQuat(refVelocity_, refQuat_, IIWA_7); }
    else if(fsm_state.mode_iiwa7 == REST){publishPosQuat(returnPos_, refQuat_, IIWA_7);}
    
    if(fsm_state.mode_iiwa14 == HIT){ publishVelQuat(refVelocity_, refQuat_, IIWA_14); }
    else if(fsm_state.mode_iiwa14 == REST){publishPosQuat(returnPos_, refQuat_, IIWA_14);}
    
    ros::spinOnce();
    rate_.sleep();
  }

  std::cout << "STOPPING AIR HOCKEY " << std::endl; 
  // publishVelQuat(refVelocity_, refQuat_);
  publishPosQuat(returnPos_, refQuat_, IIWA_7);
  publishPosQuat(returnPos_, refQuat_, IIWA_14);
  ros::spinOnce();
  rate_.sleep();
  ros::shutdown();
}

int main(int argc, char** argv) {

  //ROS Initialization
  ros::init(argc, argv, "airhockey");
  ros::NodeHandle nh;
  float frequency = 200.0f;

  std::unique_ptr<AirHockey> generate_motion = std::make_unique<AirHockey>(nh, frequency);

  if (!generate_motion->init()) {
    return -1;
  } else {
    std::cout << "OK "<< std::endl;;
    generate_motion->run();
  }

  return 0;
}
