#include "AirHockey.hpp"

bool AirHockey::init() {

  // Check if sim or real
  if (!nh_.getParam("simulation_referential",isSim_)) { ROS_ERROR("Param simulation_referential not found"); }
  // Check if automatic
  if (!nh_.getParam("automatic",isAuto_)) { ROS_ERROR("Param automatic not found"); }
  // Check if using fixed flux
  if (!nh_.getParam("fixed_flux",isFluxFixed_)) { ROS_ERROR("Param automatic not found"); }
  // Check safetz distance
  if (!nh_.getParam("safety_distance", objectSafetyDistance_)) { ROS_ERROR("Param safety distance not found"); }
  // Check tiem to wait between hits
  if (!nh_.getParam("time_to_wait",timeToWait_)) { ROS_ERROR("Param tiem to waitnot found"); }
  waitDuration_= ros::Duration(timeToWait_);

  // Get topics names
  if (!nh_.getParam("recorder_topic", pubFSMTopic_)) {ROS_ERROR("Topic /recorder/robot_states not found");}

  if (!nh_.getParam("/passive_control/vel_quat_7", pubVelQuatTopic_[IIWA_7])) {ROS_ERROR("Topic /passive_control iiwa 7 not found");}
  if (!nh_.getParam("/passive_control/vel_quat_14", pubVelQuatTopic_[IIWA_14])) {ROS_ERROR("Topic /passive_control iiwa 14 not found");}

  if (!nh_.getParam("/passive_control/pos_quat_7", pubPosQuatTopic_[IIWA_7])) {ROS_ERROR("Topic /passive_control iiwa 7 not found");}
  if (!nh_.getParam("/passive_control/pos_quat_14", pubPosQuatTopic_[IIWA_14])) {ROS_ERROR("Topic /passive_control iiwa 14 not found");}

  if (!nh_.getParam("/iiwa/inertia/taskPos_7", iiwaInertiaTopic_[IIWA_7])) {ROS_ERROR("Topic /iiwa/inertia/taskPos not found");}
  if (!nh_.getParam("/iiwa/inertia/taskPos_14", iiwaInertiaTopic_[IIWA_14])) {ROS_ERROR("Topic /iiwa/inertia/taskPos not found");}

  if(isSim_){
    if (!nh_.getParam("/gazebo/link_states", iiwaPositionTopicSim_)) {ROS_ERROR("Topic /gazebo/link_states not found");}
    if (!nh_.getParam("/gazebo/model_states", objectPositionTopic_)) {ROS_ERROR("Topic /gazebo/model_states not found");}
  }

  else if (!isSim_){
    if (!nh_.getParam("/iiwa/info_7/pose", iiwaPositionTopicReal_[IIWA_7])) {ROS_ERROR("Topic /iiwa1/ee_info/pose not found");}
    if (!nh_.getParam("/iiwa/info_14/pose", iiwaPositionTopicReal_[IIWA_14])) {ROS_ERROR("Topic /iiwa2/ee_info/pose not found");}
    if (!nh_.getParam("/iiwa/info_7/vel", iiwaVelocityTopicReal_[IIWA_7])) {ROS_ERROR("Topic /iiwa1/ee_info/vel not found");}
    if (!nh_.getParam("/iiwa/info_14/vel", iiwaVelocityTopicReal_[IIWA_14])) {ROS_ERROR("Topic /iiwa2/ee_info/vel not found");}
    if (!nh_.getParam("/vrpn_client_node/object_1/pose", objectPositionTopic_)) {ROS_ERROR("Topic vrpn/object1 not found");}
    if (!nh_.getParam("/vrpn_client_node/iiwa_7_base/pose", iiwaBasePositionTopic_[IIWA_7])) {ROS_ERROR("Topic vrpn/iiwa7 not found");}
    if (!nh_.getParam("/vrpn_client_node/iiwa_14_base/pose", iiwaBasePositionTopic_[IIWA_14])) {ROS_ERROR("Topic vrpn/iiwa14 not found");}
  }
  
  // Init publishers
  pubVelQuat_[IIWA_7] = nh_.advertise<geometry_msgs::Pose>(pubVelQuatTopic_[IIWA_7], 1);
  pubVelQuat_[IIWA_14] = nh_.advertise<geometry_msgs::Pose>(pubVelQuatTopic_[IIWA_14], 1);
  pubPosQuat_[IIWA_7] = nh_.advertise<geometry_msgs::Pose>(pubPosQuatTopic_[IIWA_7], 1);
  pubPosQuat_[IIWA_14] = nh_.advertise<geometry_msgs::Pose>(pubPosQuatTopic_[IIWA_14], 1);
  pubFSM_ = nh_.advertise<i_am_project::FSM_state>(pubFSMTopic_, 1);

  // Init subscribers
  if(isSim_){
    objectPosition_ = nh_.subscribe(objectPositionTopic_,
                                  1,
                                  &AirHockey::objectPositionCallbackGazebo,
                                  this,
                                  ros::TransportHints().reliable().tcpNoDelay());
    iiwaPosition_ = nh_.subscribe(iiwaPositionTopicSim_,
                                1,
                                &AirHockey::iiwaPositionCallbackGazebo,
                                this,
                                ros::TransportHints().reliable().tcpNoDelay());

  }
  else if (!isSim_){
    objectPosition_ = nh_.subscribe(objectPositionTopic_,
                                   1,
                                   &AirHockey::objectPositionCallbackReal,
                                   this,
                                   ros::TransportHints().reliable().tcpNoDelay());

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

    iiwaBasePosition_[IIWA_7] = 
        nh_.subscribe<geometry_msgs::PoseStamped>(iiwaBasePositionTopic_[IIWA_7],
                                                  1,
                                                  boost::bind(&AirHockey::iiwaBasePositionCallbackReal, this, _1, IIWA_7),
                                                  ros::VoidPtr(),
                                                  ros::TransportHints().reliable().tcpNoDelay());
    
    iiwaBasePosition_[IIWA_14] = 
        nh_.subscribe<geometry_msgs::PoseStamped>(iiwaBasePositionTopic_[IIWA_14],
                                                  1,
                                                  boost::bind(&AirHockey::iiwaBasePositionCallbackReal, this, _1, IIWA_14),
                                                  ros::VoidPtr(),
                                                  ros::TransportHints().reliable().tcpNoDelay());
  }
  
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

  while (objectPositionFromSource_.norm() == 0) {
    this->updateDSAttractor();
    ros::spinOnce();
    rate_.sleep();
  }

  generateHitting7_->set_current_position(iiwaPositionFromSource_[IIWA_7]);
  generateHitting14_->set_current_position(iiwaPositionFromSource_[IIWA_14]);
  this->updateDSAttractor(); //update attractor position

  // Get hitting parameters
  if (!nh_.getParam("iiwa7/ref_velocity/y", refVelocity_[IIWA_7][1])) { ROS_ERROR("Param ref_velocity/y not found"); }
  if (!nh_.getParam("iiwa14/ref_velocity/y", refVelocity_[IIWA_14][1])) { ROS_ERROR("Param ref_velocity/y not found"); }
  if (!nh_.getParam("iiwa7/ref_velocity/x", refVelocity_[IIWA_7][0])) { ROS_ERROR("Param ref_velocity/x not found"); }
  if (!nh_.getParam("iiwa14/ref_velocity/x", refVelocity_[IIWA_14][0])) { ROS_ERROR("Param ref_velocity/x not found"); }
  if (!nh_.getParam("iiwa7/ref_velocity/z", refVelocity_[IIWA_7][2])) { ROS_ERROR("Param ref_velocity/z not found"); }
  if (!nh_.getParam("iiwa14/ref_velocity/z", refVelocity_[IIWA_14][2])) { ROS_ERROR("Param ref_velocity/z not found"); }
  if (!nh_.getParam("iiwa7/ref_orientation/w", refQuat_[IIWA_7][0])) { ROS_ERROR("Param ref_orientation/w not found"); }
  if (!nh_.getParam("iiwa14/ref_orientation/w", refQuat_[IIWA_14][0])) { ROS_ERROR("Param ref_orientation/w not found"); }
  if (!nh_.getParam("iiwa7/ref_orientation/x", refQuat_[IIWA_7][1])) { ROS_ERROR("Param ref_orientation/x not found"); }
  if (!nh_.getParam("iiwa14/ref_orientation/x", refQuat_[IIWA_14][1])) { ROS_ERROR("Param ref_orientation/x not found"); }
  if (!nh_.getParam("iiwa7/ref_orientation/y", refQuat_[IIWA_7][2])) { ROS_ERROR("Param ref_orientation/y not found"); }
  if (!nh_.getParam("iiwa14/ref_orientation/y", refQuat_[IIWA_14][2])) { ROS_ERROR("Param ref_orientation/y not found"); }
  if (!nh_.getParam("iiwa7/ref_orientation/z", refQuat_[IIWA_7][3])) { ROS_ERROR("Param ref_orientation/z not found"); }
  if (!nh_.getParam("iiwa14/ref_orientation/z", refQuat_[IIWA_14][3])) { ROS_ERROR("Param ref_orientation/z not found"); }
  if (!nh_.getParam("iiwa7/return_position/x", returnPos_[IIWA_7][0])) { ROS_ERROR("Param ref_quat/x not found"); }
  if (!nh_.getParam("iiwa14/return_position/x", returnPos_[IIWA_14][0])) { ROS_ERROR("Param return_position/x not found"); }
  if (!nh_.getParam("iiwa7/return_position/y", returnPos_[IIWA_7][1])) { ROS_ERROR("Param return_position/y not found"); }
  if (!nh_.getParam("iiwa14/return_position/y", returnPos_[IIWA_14][1])) { ROS_ERROR("Param return_position/y not found"); }
  if (!nh_.getParam("iiwa7/return_position/z", returnPos_[IIWA_7][2])) { ROS_ERROR("Param return_position/z not found"); }
  if (!nh_.getParam("iiwa14/return_position/z", returnPos_[IIWA_14][2])) { ROS_ERROR("Param return_position/z not found"); }
  if (!nh_.getParam("iiwa7/hit_direction/x", hitDirection_[IIWA_7][0])) {ROS_ERROR("Param hit_direction/x not found");}
  if (!nh_.getParam("iiwa14/hit_direction/x", hitDirection_[IIWA_14][0])) {ROS_ERROR("Param hit_direction/x not found");}
  if (!nh_.getParam("iiwa7/hit_direction/y", hitDirection_[IIWA_7][1])) {ROS_ERROR("Param hit_direction/y not found");}
  if (!nh_.getParam("iiwa14/hit_direction/y", hitDirection_[IIWA_14][1])) {ROS_ERROR("Param hit_direction/y not found");}
  if (!nh_.getParam("iiwa7/hit_direction/z", hitDirection_[IIWA_7][2])) {ROS_ERROR("Param hit_direction/z not found");}
  if (!nh_.getParam("iiwa14/hit_direction/z", hitDirection_[IIWA_14][2])) {ROS_ERROR("Param hit_direction/z not found");}
  if (!nh_.getParam("iiwa7/hitting_flux", hittingFlux_[IIWA_7])) {ROS_ERROR("Param iiwa7/hitting_flux not found");}
  if (!nh_.getParam("iiwa14/hitting_flux", hittingFlux_[IIWA_14])) {ROS_ERROR("Param iiwa14/hitting_flux not found");}
  if (!nh_.getParam("object_mass", objectMass_)) {ROS_ERROR("Param object_mass not found");}

  generateHitting7_->set_des_direction(hitDirection_[IIWA_7]);
  generateHitting14_->set_des_direction(hitDirection_[IIWA_14]);

  // Initialize PAUSE state
  isPaused_ = true;
  next_hit_ = IIWA_7;

  // add gazebo offset to calculate norm correctly in Sim
  if(isSim_){
    returnPosGazebo_[IIWA_7] = returnPos_[IIWA_7];
    returnPosGazebo_[IIWA_14] = returnPos_[IIWA_14];
    returnPosGazebo_[IIWA_7][1] = returnPos_[IIWA_7][1] - 0.6;
    returnPosGazebo_[IIWA_14][1] = returnPos_[IIWA_14][1] + 0.6;
  }

  // get desired hitting fluxes from file in config folder
  if(!isFluxFixed_){
    std::string flux_fn;
    if (!nh_.getParam("desired_fluxes_filename",flux_fn)) { ROS_ERROR("Param desired_fluxes_filename not found"); }

    flux_fn = "/home/ros/ros_ws/src/i_am_project/desired_hitting_fluxes/" + flux_fn;
    this->getDesiredFluxes(flux_fn);

    hittingFlux_[IIWA_7] = hittingFluxArr_[0];
    hittingFlux_[IIWA_14] = hittingFluxArr_[0];
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
  iiwaOrientationFromSource_[IIWA_7] << iiwaPose_[IIWA_7].orientation.w, iiwaPose_[IIWA_7].orientation.x, 
      iiwaPose_[IIWA_7].orientation.y, iiwaPose_[IIWA_7].orientation.z;    

  iiwaVel_[IIWA_7] = linkStates.twist[indexIiwa1];
  iiwaVelocityFromSource_[IIWA_7] << iiwaVel_[IIWA_7].linear.x, iiwaVel_[IIWA_7].linear.y,
      iiwaVel_[IIWA_7].linear.z;

  iiwaPose_[IIWA_14] = linkStates.pose[indexIiwa2];
  iiwaPositionFromSource_[IIWA_14] << iiwaPose_[IIWA_14].position.x, iiwaPose_[IIWA_14].position.y,
      iiwaPose_[IIWA_14].position.z;
  iiwaOrientationFromSource_[IIWA_14] << iiwaPose_[IIWA_14].orientation.w, iiwaPose_[IIWA_14].orientation.x, 
      iiwaPose_[IIWA_14].orientation.y, iiwaPose_[IIWA_14].orientation.z; 

  iiwaVel_[IIWA_14] = linkStates.twist[indexIiwa2];
  iiwaVelocityFromSource_[IIWA_14] << iiwaVel_[IIWA_14].linear.x, iiwaVel_[IIWA_14].linear.y,
      iiwaVel_[IIWA_14].linear.z;
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

void AirHockey::iiwaBasePositionCallbackReal(const geometry_msgs::PoseStamped::ConstPtr& msg, int k){
  iiwaBasePositionFromSource_[k]  << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
}

void AirHockey::objectPositionCallbackReal(const geometry_msgs::PoseStamped::ConstPtr& msg){
  objectPositionFromSource_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  objectOrientationFromSource_ << msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z,
      msg->pose.orientation.w;
}

// UPDATES AND CALCULATIONS
void AirHockey::objectPositionIiwaFrames() {
  rotationMat_ << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;

  objectPositionForIiwa_[IIWA_7] = rotationMat_ * (objectPositionFromSource_ - iiwaBasePositionFromSource_[IIWA_7]);
  objectPositionForIiwa_[IIWA_14] = rotationMat_ * (objectPositionFromSource_ - iiwaBasePositionFromSource_[IIWA_14]);
}

void AirHockey::updateDSAttractor() {

  if(isSim_){
    generateHitting7_->set_DS_attractor(objectPositionFromSource_);
    generateHitting14_->set_DS_attractor(objectPositionFromSource_);
  }
  else if(!isSim_){
    objectPositionIiwaFrames();

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
    if(objectPositionFromSource_.norm() > objectSafetyDistance_){
      isPaused_ = true;
    }
    // need condition to unPause if object enters zone 
    // TODO : save old norm and compare?
  }
  else if(!isSim_){
    float object_safety_threshold = 1.0;

    if((objectPositionForIiwa_[IIWA_7].norm() > object_safety_threshold) && (objectPositionForIiwa_[IIWA_14].norm() > object_safety_threshold)){
      isPaused_ = true;
    }  
  }
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
  i_am_project::FSM_state msg;
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
        current_state.mode_iiwa7 = HIT;
        std::cout << "q is pressed " << std::endl;
        
      } break;
      case 'p': {
        current_state.mode_iiwa14 = HIT;
        std::cout << "p is pressed " << std::endl;
      } break;
      case 'r': {
        current_state.mode_iiwa7 = REST;
        current_state.mode_iiwa14 = REST;
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
    current_state.mode_iiwa7 = REST;
    current_state.mode_iiwa14 = REST;
    return current_state;
  }

  // if AUTO -> check robots are ready to hit then set to hit
  float object_stopped_threshold = 2*1e-4;
  float iiwa_at_rest_threshold = 5*1e-2;
  float norm_object = (previousObjectPositionFromSource_-objectPositionFromSource_).norm();

  // Only set to HIt if both robots are at rest!
  if(current_state.mode_iiwa7 == REST && current_state.mode_iiwa14 == REST){
    
    // Stay in REST for desired duration
    auto time_in_rest = ros::Time::now() - current_state.hit_time;
    if(time_in_rest < waitDuration_){ 
      return current_state;
    }

    // is object stopped?
    if(norm_object < object_stopped_threshold){
      
      // Get norms
      float norm_iiwa7 = (iiwaPositionFromSource_[IIWA_7]-returnPos_[IIWA_7]).norm();
      float norm_iiwa14 = (iiwaPositionFromSource_[IIWA_14]-returnPos_[IIWA_14]).norm();
      // re-write if in simulation
      if(isSim_){
        norm_iiwa7 = (iiwaPositionFromSource_[IIWA_7]-returnPosGazebo_[IIWA_7]).norm();
        norm_iiwa14 = (iiwaPositionFromSource_[IIWA_14]-returnPosGazebo_[IIWA_14]).norm();}

      // are robots at rest positions ?
      if(norm_iiwa7 < iiwa_at_rest_threshold && norm_iiwa14 < iiwa_at_rest_threshold){
        // Then set to HIT and update next_hit
        if(next_hit_ == IIWA_7){
          current_state.mode_iiwa7 = HIT;
          next_hit_ = IIWA_14;
        }
        else if(next_hit_ == IIWA_14){
          current_state.mode_iiwa14 = HIT;
          next_hit_ = IIWA_7;
        }
      }
    }
  }

  // Update previous object pos
  previousObjectPositionFromSource_ = objectPositionFromSource_;

  return current_state;
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
    if(!isAuto_) {fsm_state = updateKeyboardControl(fsm_state); }
    // Otherwise update FSM automatically
    else if (isAuto_)
    {
      // function call to check keyboard, see if it is paused
      updateisPaused();

      // check if object is within range (for safety)
      checkObjectIsSafeToHit();

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
      // std::cout << "iiwa7_state : " << fsm_state.mode_iiwa7 << " \n iiwa14_state : " << fsm_state.mode_iiwa14<< std::endl;
      // std::cout << "object source pos  " << objectPositionFromSource_ << std::endl;
      // std::cout << "iiwaPos_7  " << iiwaBasePositionFromSource_[IIWA_7]<< std::endl;
      // std::cout << "iiwaPos_14  " << iiwaBasePositionFromSource_[IIWA_14]<< std::endl;
      // std::cout << "object pos by iiwaPos_7  " << objectPositionForIiwa_[IIWA_7]<< std::endl;
      // std::cout << "object pos by  iiwaPos_14  " << objectPositionForIiwa_[IIWA_14]<< std::endl;
      // std::cout << "returnPos_7  " << returnPosGazebo_[IIWA_7]<< std::endl;
      // std::cout << "returnPos_14  " << returnPosGazebo_[IIWA_14]<< std::endl; //objectPositionForIiwa_[IIWA_7]
    }
    print_count +=1 ;

    // Update DS attractor if at rest
    if(fsm_state.mode_iiwa7 == REST && fsm_state.mode_iiwa14 == REST){
      // update only at REST so object state conditions for isHit works 
      updateDSAttractor();
    }

    // UPDATE robot state
    if(fsm_state.mode_iiwa7 == HIT){
      refVelocity_[IIWA_7] = generateHitting7_->flux_DS(hittingFlux_[IIWA_7], iiwaTaskInertiaPosInv_[IIWA_7]);
    }

    if(fsm_state.mode_iiwa14 == HIT){
      refVelocity_[IIWA_14] = generateHitting14_->flux_DS(hittingFlux_[IIWA_14], iiwaTaskInertiaPosInv_[IIWA_14]);
      update_flux_once = 1; // only update after 1 hit from each robot
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

    updateCurrentEEPosition(iiwaPositionFromSource_);

    // Publisher logic to use publish position when returning (avoids inertia in iiwa_toolkit)
    if(fsm_state.mode_iiwa7 == HIT){ publishVelQuat(refVelocity_, refQuat_, IIWA_7); }
    else if(fsm_state.mode_iiwa7 == REST){publishPosQuat(returnPos_, refQuat_, IIWA_7);}
    
    if(fsm_state.mode_iiwa14 == HIT){ publishVelQuat(refVelocity_, refQuat_, IIWA_14); }
    else if(fsm_state.mode_iiwa14 == REST){publishPosQuat(returnPos_, refQuat_, IIWA_14);}
    
    ros::spinOnce();
    rate_.sleep();
  }

  // publishVelQuat(refVelocity_, refQuat_);
  publishPosQuat(returnPos_, refQuat_, IIWA_7);
  publishPosQuat(returnPos_, refQuat_, IIWA_14);
  ros::spinOnce();
  rate_.sleep();
  ros::shutdown();
}

int main(int argc, char** argv) {

  //ROS Initialization
  ros::init(argc, argv, "air_hockey");
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
