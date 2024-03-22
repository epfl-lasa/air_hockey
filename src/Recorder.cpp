#include "Recorder.hpp"

bool Recorder::init() {

  // Check if sim or real
  if (!nh_.getParam("simulation_referential",isSim_)) { ROS_ERROR("Param simulation not found"); }

  // check if recording
  if (!nh_.getParam("recording",isRecording_)) { ROS_ERROR("Param recording not found"); }
  if (!nh_.getParam("recorder_folder", recordingFolderPath_)) { ROS_ERROR("Param recorder_path not found"); }
  if (!nh_.getParam("time_object_record", recordingTimeObject_)) { ROS_ERROR("Param recorder_time not found"); }
  if (!nh_.getParam("post_hit_record", recordingTimeRobot_)) { ROS_ERROR("Param recorder_time not found"); }

  if (!nh_.getParam("object_mass", objectMass_)) {ROS_ERROR("Param object_mass not found");}

  if (!nh_.getParam("desired_fluxes_filename",fluxFilename_)) { ROS_ERROR("Param automatic not found"); }

  // Get topics names
  if (!nh_.getParam("recorder_topic", FSMTopic_)) {ROS_ERROR("Topic /recorder/robot_states not found");}

  if (!nh_.getParam("/iiwa/inertia/taskPos_7", iiwaInertiaTopic_[IIWA_7])) {ROS_ERROR("Topic /iiwa/inertia/taskPos not found");}
  if (!nh_.getParam("/iiwa/inertia/taskPos_14", iiwaInertiaTopic_[IIWA_14])) {ROS_ERROR("Topic /iiwa/inertia/taskPos not found");}

  if (!nh_.getParam("/passive_control/vel_quat_7", pubVelQuatTopic_[IIWA_7])) {ROS_ERROR("Topic /passive_control iiwa 7 not found");}
  if (!nh_.getParam("/passive_control/vel_quat_14", pubVelQuatTopic_[IIWA_14])) {ROS_ERROR("Topic /passive_control iiwa 14 not found");}

  if (!nh_.getParam("/iiwa/info_7/joint_state", iiwaJointStateTopicReal_[IIWA_7])) {ROS_ERROR("Topic /iiwa1/joint_state not found");}
  if (!nh_.getParam("/iiwa/info_14/joint_state", iiwaJointStateTopicReal_[IIWA_14])) {ROS_ERROR("Topic /iiwa2/joint_state not found");}
  if (!nh_.getParam("/iiwa/info_7/trq_cmd", iiwaTorqueCmdTopic_[IIWA_7])) {ROS_ERROR("Topic /iiwa1/TorqueController/command not found");}
  if (!nh_.getParam("/iiwa/info_14/trq_cmd", iiwaTorqueCmdTopic_[IIWA_14])) {ROS_ERROR("Topic /iiwa2/TorqueController/command not found");}

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
  }

  // Init subscribers
  if(isSim_){
    objectPosition_ = nh_.subscribe(objectPositionTopic_,
                                  1,
                                  &Recorder::objectPositionCallbackGazebo,
                                  this,
                                  ros::TransportHints().reliable().tcpNoDelay());
    iiwaPosition_ = nh_.subscribe(iiwaPositionTopicSim_,
                                1,
                                &Recorder::iiwaPositionCallbackGazebo,
                                this,
                                ros::TransportHints().reliable().tcpNoDelay());

  }
  else if (!isSim_){
    objectPosition_ = nh_.subscribe(objectPositionTopic_,
                                   1,
                                   &Recorder::objectPositionCallbackReal,
                                   this,
                                   ros::TransportHints().reliable().tcpNoDelay());

    iiwaPositionReal_[IIWA_7] = 
        nh_.subscribe<geometry_msgs::Pose>(iiwaPositionTopicReal_[IIWA_7],
                                            1,
                                            boost::bind(&Recorder::iiwaPoseCallbackReal, this, _1, IIWA_7),
                                            ros::VoidPtr(),
                                            ros::TransportHints().reliable().tcpNoDelay());    

    iiwaPositionReal_[IIWA_14] = 
        nh_.subscribe<geometry_msgs::Pose>(iiwaPositionTopicReal_[IIWA_14],
                                            1,
                                            boost::bind(&Recorder::iiwaPoseCallbackReal, this, _1, IIWA_14),
                                            ros::VoidPtr(),
                                            ros::TransportHints().reliable().tcpNoDelay());
 
  iiwaVelocityReal_[IIWA_7] = 
      nh_.subscribe<geometry_msgs::Twist>(iiwaVelocityTopicReal_[IIWA_7],
                                          1,
                                          boost::bind(&Recorder::iiwaVelocityCallbackReal, this, _1, IIWA_7),
                                          ros::VoidPtr(),
                                          ros::TransportHints().reliable().tcpNoDelay());    

  iiwaVelocityReal_[IIWA_14] = 
      nh_.subscribe<geometry_msgs::Twist>(iiwaVelocityTopicReal_[IIWA_14],
                                          1,
                                          boost::bind(&Recorder::iiwaVelocityCallbackReal, this, _1, IIWA_14),
                                          ros::VoidPtr(),
                                          ros::TransportHints().reliable().tcpNoDelay());
  }

  iiwaJointStateReal_[IIWA_7] = 
      nh_.subscribe<sensor_msgs::JointState>(iiwaJointStateTopicReal_[IIWA_7],
                                          1,
                                          boost::bind(&Recorder::iiwaJointStateCallbackReal, this, _1, IIWA_7),
                                          ros::VoidPtr(),
                                          ros::TransportHints().reliable().tcpNoDelay());    

  iiwaJointStateReal_[IIWA_14] = 
      nh_.subscribe<sensor_msgs::JointState>(iiwaJointStateTopicReal_[IIWA_14],
                                          1,
                                          boost::bind(&Recorder::iiwaJointStateCallbackReal, this, _1, IIWA_14),
                                          ros::VoidPtr(),
                                          ros::TransportHints().reliable().tcpNoDelay());

  iiwaInertia_[IIWA_7] =
      nh_.subscribe<geometry_msgs::Inertia>(iiwaInertiaTopic_[IIWA_7],
                                            1,
                                            boost::bind(&Recorder::iiwaInertiaCallback, this, _1, IIWA_7),
                                            ros::VoidPtr(),
                                            ros::TransportHints().reliable().tcpNoDelay());
  iiwaInertia_[IIWA_14] =
      nh_.subscribe<geometry_msgs::Inertia>(iiwaInertiaTopic_[IIWA_14],
                                            1,
                                            boost::bind(&Recorder::iiwaInertiaCallback, this, _1, IIWA_14),
                                            ros::VoidPtr(),
                                            ros::TransportHints().reliable().tcpNoDelay());

  iiwaDesiredVelocity_[IIWA_7] = 
        nh_.subscribe<geometry_msgs::Pose>(pubVelQuatTopic_[IIWA_7],
                                            1,
                                            boost::bind(&Recorder::iiwaDesiredVelocityCallback, this, _1, IIWA_7),
                                            ros::VoidPtr(),
                                            ros::TransportHints().reliable().tcpNoDelay());    

  iiwaDesiredVelocity_[IIWA_14] = 
        nh_.subscribe<geometry_msgs::Pose>(pubVelQuatTopic_[IIWA_14],
                                            1,
                                            boost::bind(&Recorder::iiwaDesiredVelocityCallback, this, _1, IIWA_14),
                                            ros::VoidPtr(),
                                            ros::TransportHints().reliable().tcpNoDelay());
  iiwaTorqueCmd_[IIWA_7] = 
      nh_.subscribe<std_msgs::Float64MultiArray>(iiwaTorqueCmdTopic_[IIWA_7],
                                          1,
                                          boost::bind(&Recorder::iiwaTorqueCmdCallback, this, _1, IIWA_7),
                                          ros::VoidPtr(),
                                          ros::TransportHints().reliable().tcpNoDelay());    

  iiwaTorqueCmd_[IIWA_14] = 
      nh_.subscribe<std_msgs::Float64MultiArray>(iiwaTorqueCmdTopic_[IIWA_14],
                                          1,
                                          boost::bind(&Recorder::iiwaTorqueCmdCallback, this, _1, IIWA_14),
                                          ros::VoidPtr(),
                                          ros::TransportHints().reliable().tcpNoDelay());

  FSMState_ = nh_.subscribe(FSMTopic_,
                            1,
                            &Recorder::FSMCallback,
                            this,
                            ros::TransportHints().reliable().tcpNoDelay());

  previousObjectPositionFromSource_ = objectPositionFromSource_; // set prev position
  moved_manually_count_ = 1; // set count for moved manually

  // resize once here to avoid doing it in subscriber 
  iiwaTorqueCmdFromSource_[IIWA_7].resize(7);
  iiwaTorqueCmdFromSource_[IIWA_14].resize(7);

  return true;
}

// CALLBACKS
void Recorder::objectPositionCallbackGazebo(const gazebo_msgs::ModelStates& modelStates) {
  int boxIndex = getIndex(modelStates.name, "box_model");
  boxPose_ = modelStates.pose[boxIndex];
  objectPositionFromSource_ << boxPose_.position.x, boxPose_.position.y, boxPose_.position.z;
  objectOrientationFromSource_ << boxPose_.orientation.x, boxPose_.orientation.y, boxPose_.orientation.z,
      boxPose_.orientation.w;
}

void Recorder::iiwaPositionCallbackGazebo(const gazebo_msgs::LinkStates& linkStates) {
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

void Recorder::iiwaInertiaCallback(const geometry_msgs::Inertia::ConstPtr& msg, int k) {
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

void Recorder::iiwaPoseCallbackReal(const geometry_msgs::Pose::ConstPtr& msg, int k){
  iiwaPositionFromSource_[k]  << msg->position.x, msg->position.y, msg->position.z;
  iiwaOrientationFromSource_[k] << msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z;
}

void Recorder::iiwaVelocityCallbackReal(const geometry_msgs::Twist::ConstPtr& msg, int k){
  iiwaVelocityFromSource_[k]  << msg->linear.x, msg->linear.y, msg->linear.z;
}

void Recorder::objectPositionCallbackReal(const geometry_msgs::PoseStamped::ConstPtr& msg){
  objectPositionFromSource_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  objectOrientationFromSource_ << msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z;
}

void Recorder::iiwaJointStateCallbackReal(const sensor_msgs::JointState::ConstPtr& msg, int k){
  iiwaJointState_[k].name =  msg->name;
  iiwaJointState_[k].position =  msg->position;
  iiwaJointState_[k].velocity =  msg->velocity;
  iiwaJointState_[k].effort =  msg->effort; 
}

void Recorder::iiwaDesiredVelocityCallback(const geometry_msgs::Pose::ConstPtr& msg, int k){
  iiwaDesiredVelocityFromSource_[k]  << msg->position.x, msg->position.y, msg->position.z;
}

void Recorder::iiwaTorqueCmdCallback(const std_msgs::Float64MultiArray::ConstPtr &msg, int k){
  
  size_t num_elements = msg->data.size();

  for (size_t i = 0; i < num_elements; ++i) {
      iiwaTorqueCmdFromSource_[k](i) = msg->data[i];
  }
}

void Recorder::FSMCallback(const i_am_project::FSM_state::ConstPtr& msg){
  fsmState_.mode_iiwa7 = static_cast<robotMode>(msg->mode_iiwa7);
  fsmState_.mode_iiwa14 = static_cast<robotMode>(msg->mode_iiwa14);
  fsmState_.isHit = msg->isHit;
  fsmState_.hit_time = msg->hit_time;
  fsmState_.des_flux = msg->des_flux;
  fsmState_.des_pos(0) = msg->des_pos_x; 
  fsmState_.des_pos(1) = msg->des_pos_y; 
  fsmState_.des_pos(2) = msg->des_pos_z; 
}

// UPDATES AND CALCULATIONS
int Recorder::getIndex(std::vector<std::string> v, std::string value) {
  for (int i = 0; i < v.size(); i++) {
    if (v[i].compare(value) == 0) return i;
  }
  return -1;
}

float Recorder::calculateDirFlux(Robot robot_name) {
  return ((1/iiwaTaskInertiaPosInv_[robot_name](1, 1)) / (1/(iiwaTaskInertiaPosInv_[robot_name](1, 1)) + objectMass_)) * iiwaVelocityFromSource_[robot_name].norm();
}

// RECORDING FUNCTIONS 
std::string Recorder::robotToString(Robot robot_name) {
  switch (robot_name) {
      case IIWA_7:
          return "IIWA_7";
      case IIWA_14:
          return "IIWA_14";
      default:
          return "Unknown Robot";
  }
}

void Recorder::recordRobot(Robot robot_name){
  // record robot data during HIT phase
  // joint state (pos,vel,effort), torque command, EEF position + velocity, Desired pos+vel+flux, inertia, hitting flux

  RecordedRobotState newState;
  newState.robot_name = robotToString(robot_name);
  // Get the current time
  newState.time = ros::Time::now();

  newState.joint_pos.resize(7);
  Eigen::Map<Eigen::VectorXd> tempVector1(iiwaJointState_[robot_name].position.data(), iiwaJointState_[robot_name].position.size());
  newState.joint_pos = tempVector1;

  newState.joint_vel.resize(7);
  Eigen::Map<Eigen::VectorXd> tempVector2(iiwaJointState_[robot_name].velocity.data(), iiwaJointState_[robot_name].velocity.size());
  newState.joint_vel =  tempVector2;

  newState.joint_effort.resize(7);
  Eigen::Map<Eigen::VectorXd> tempVector3(iiwaJointState_[robot_name].effort.data(), iiwaJointState_[robot_name].effort.size());
  newState.joint_effort =  tempVector3;

  newState.trq_cmd = iiwaTorqueCmdFromSource_[robot_name];

  newState.eef_pos.resize(3);
  newState.eef_pos = iiwaPositionFromSource_[robot_name];

  newState.eef_orientation.resize(4);
  newState.eef_orientation = iiwaOrientationFromSource_[robot_name];

  newState.eef_vel.resize(3);
  newState.eef_vel = iiwaVelocityFromSource_[robot_name];

  newState.eef_vel_des.resize(3);
  newState.eef_vel_des = iiwaDesiredVelocityFromSource_[robot_name];

  newState.inertia.resize(9);
  Eigen::Map<Eigen::Matrix<float, 9, 1>> tempVector4(iiwaTaskInertiaPosInv_[robot_name].data());
  newState.inertia = tempVector4;

  newState.hitting_flux = calculateDirFlux(robot_name);

  // Add the new state to the vector
  robotStatesVector_[robot_name].push_back(newState);

}

void Recorder::recordObject(bool manual){
  // record object position 
  // Start when robot enters HIT phase, ends with a timer after X seconds

  RecordedObjectState newState;
  // Get the current time
  newState.time = ros::Time::now();
  newState.position = objectPositionFromSource_;
  newState.orientation = objectOrientationFromSource_;

  // Add the new state to the vector
  if(manual){objectStatesVectorManual_.push_back(newState);}
  else if(!manual){objectStatesVector_.push_back(newState);}
}

void Recorder::recordObjectMovedByHand(int hit_count){
  // Called when both robots are at rest
  // Detects if object is moving and record if so. Writes to file when stops moving

  bool manual = true;
  float stopped_threshold = 2*1e-4;
  float moving_threshold = 1e-3;
  float norm = (previousObjectPositionFromSource_-objectPositionFromSource_).norm();
  if(norm == 0){return;} // object callback has not yet been updated
  // Need to update prevPosition -> we consider object cannot manually be moved more than 10cm in 5ms
  if(norm > 0.1){norm = 0;}// hack to avoid writing file right after hit 

  if((norm < stopped_threshold) && isObjectMoving_){
    // was moving but stopped -> write to file
    std::string fn = recordingFolderPath_ + "object_moved_manually_after_hit_"+ std::to_string(hit_count)+"-"+std::to_string(moved_manually_count_)+".csv";
    writeObjectStatesToFile(hit_count, fn, manual);
    moved_manually_count_ += 1;
    isObjectMoving_ = 0;
    std::cout << "Finished writing motion for object moved manually after hit " << std::to_string(hit_count) << "-" << std::to_string(moved_manually_count_) << std::endl;
  }
  else if(norm < stopped_threshold){
    // not moving -> do nothing
    isObjectMoving_ = 0;
  }
  else if(norm > moving_threshold){
    // moving -> record object
    isObjectMoving_ = 1;
    recordObject(manual);
  }

  previousObjectPositionFromSource_ = objectPositionFromSource_;
}

void Recorder::writeRobotStatesToFile(Robot robot_name, int hit_count) {
    
    std::string filename = recordingFolderPath_ + robotToString(robot_name) +"_hit_"+ std::to_string(hit_count)+".csv";
    std::ofstream outFile(filename, std::ios::app); // Open file in append mode

    if(!outFile){
        std::cerr << "Error opening file: " << filename << std::endl;
        perror("Error");
        return;
    }

    // Write single value info in first row
    outFile << "DesiredFlux," << hittingFluxDes_[robot_name] << ","
            << "HitTime," <<std::setprecision(std::numeric_limits<double>::max_digits10) << hittingTime_[robot_name].toSec() + 3600  << ","
            << "DesiredPos," << desiredPosition_[robot_name].transpose() << "\n";

    // Write CSV header
    outFile << "RobotName,RosTime,JointPosition,JointVelocity,JointEffort,TorqueCmd,EEF_Position,EEF_Orientation,EEF_Velocity,EEF_DesiredVelocity,Inertia,HittingFlux\n";

    // Write each RobotState structure to the file
    for (const auto& state : robotStatesVector_[robot_name]) {
        // Write CSV row
        outFile << state.robot_name << ","
                << std::setprecision(std::numeric_limits<double>::max_digits10) << state.time.toSec() + 3600 << "," // add precision and 1h for GMT
                << state.joint_pos.transpose() << ","
                << state.joint_vel.transpose() << ","
                << state.joint_effort.transpose() << ","
                << state.trq_cmd.transpose() << ","
                << state.eef_pos.transpose() << ","
                << state.eef_orientation.transpose() << ","
                << state.eef_vel.transpose() << ","
                << state.eef_vel_des.transpose() << ","
                << state.inertia.transpose() << ","
                << state.hitting_flux << "\n";
    }

    outFile.close();

    robotStatesVector_[robot_name].clear(); // clear vector data

    std::cout << "Finished writing hit " << hit_count << " for "<< robotToString(robot_name) << std::endl;
    
}

void Recorder::writeObjectStatesToFile(int hit_count, std::string filename, bool manual) {

  std::ofstream outFile(filename, std::ios::app); // Open file in append mode

  if(!outFile){
      std::cerr << "Error opening file: " << filename << std::endl;
      return;
  }

  // Write CSV header
  outFile << "RosTime,Position,Orientation\n";

  if(!manual){
    // Write each RobotState structure to the file
    for (const auto& state : objectStatesVector_) {
        // outFile << "Object Name: " << state.robot_name << "\n";
        outFile << std::setprecision(std::numeric_limits<double>::max_digits10) << state.time.toSec()+3600 << "," // add precision and 1h for GMT
                << state.position.transpose() << ","
                << state.orientation.transpose() << "\n";
    }

    outFile.close();

    objectStatesVector_.clear(); // clear vector data
  }

  else if(manual){
        // Write each RobotState structure to the file
    for (const auto& state : objectStatesVectorManual_) {
        // outFile << "Object Name: " << state.robot_name << "\n";
        outFile << std::setprecision(std::numeric_limits<double>::max_digits10) << state.time.toSec()+3600 << "," // add precision and 1h for GMT
                << state.position.transpose() << "\n";
    }

    outFile.close();

    objectStatesVectorManual_.clear(); // clear vector data
  }
}

void Recorder::copyYamlFile(std::string inFilePath, std::string outFilePath){
  // Open the original YAML file for reading
  std::ifstream inFile(inFilePath);

  if (!inFile.is_open()) {
      std::cerr << "Error opening original YAML file for reading\n";
      perror("Error");
      return;
  }

  // Open a new file for writing (copying)
  std::ofstream outFile(outFilePath);

  if (!outFile.is_open()) {
      std::cerr << "Error opening new YAML file for writing\n";
      perror("Error");
      return;
  }

  // Read and copy each line from the original file to the new file
  std::string line;
  while (std::getline(inFile, line)) {
      outFile << line << "\n";
  }

  // Close the files
  inFile.close();
  outFile.close();
}

void Recorder::setUpRecordingDir(){
  // Set up data structure here
  // data/Recorder/TimeStampedDir/

  // create Recorder dir
  if (mkdir(recordingFolderPath_.c_str(), S_IRWXU | S_IRWXG | S_IRWXO) == 0) {
      std::cout << "Directory created successfully: " << recordingFolderPath_ << std::endl;
  } else {
      std::cerr << "Error creating airhockey directory." << std::endl;
      perror("Error");
  }

  // giving writing authorization to recording directory -- NOT needed 
  // if (chmod(recordingFolderPath_.c_str(), S_IRWXU | S_IRWXG | S_IRWXO) == 0) {
  //     std::cout << "Write permissions set successfully for the directory." << std::endl;
  // } else {
  //     std::cerr << "Error setting write permissions for the directory." << std::endl;
  //     perror("Error");
  //     return;
  // }

  // Create Time Stamped Directory
  // Get the current time
  auto now = std::chrono::system_clock::now();

  // Convert the current time to a string
  std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
  currentTime += 3600; // add 1h for GMT correct time (due to docker not having correct timezone)
  std::tm* timeinfo = std::localtime(&currentTime);
  std::stringstream ss;
  ss << std::put_time(timeinfo, "%Y-%m-%d_%H:%M:%S");
  std::string timestamp = ss.str();

  // Create the directory
  std::string directoryPath = recordingFolderPath_ + timestamp +"/";
  if (mkdir(directoryPath.c_str(), S_IRWXU | S_IRWXG | S_IRWXO) == 0) {
      std::cout << "Directory created successfully: " << directoryPath << std::endl;
      recordingFolderPath_ = directoryPath; // Update recording path
  } else {
      std::cerr << "Error creating directory." << std::endl;
      perror("Error");
      directoryPath.clear(); // Return an empty string on error
  }

  // Copy hit_properties_air_hockey.yaml to data file to save params of recorded data 
  std::string hit_params_to_copy = "/home/ros/ros_ws/src/i_am_project/config/hit_properties_air_hockey.yaml";
  std::string hitting_params_fn = recordingFolderPath_ + "hitting_params.yaml";
  std::string toolkit_params_to_copy = "/home/ros/ros_ws/src/iiwa_toolkit_ns/config/passive_track_params_dual_real.yaml";
  std::string toolkit_params_fn = recordingFolderPath_ + "toolkit_params.yaml";
  std::string desired_fluxes_to_copy = "/home/ros/ros_ws/src/i_am_project/desired_hitting_fluxes/" + fluxFilename_;
  std::string desired_fluxes_fn = recordingFolderPath_ + "desired_hitting_fluxes-" + fluxFilename_;

  copyYamlFile(hit_params_to_copy, hitting_params_fn);
  copyYamlFile(toolkit_params_to_copy, toolkit_params_fn);
  copyYamlFile(desired_fluxes_to_copy, desired_fluxes_fn);
}


void Recorder::run() {

  // Set up recording directory structure
  if(isRecording_){setUpRecordingDir();}

  // Set up counters and bool variables
  int print_count = 0;
  int hit_count = 1;
  bool write_once_7 = 0;
  bool write_once_14 = 0;
  bool write_once_object = 0;
  bool manual = false; // only call functin to record hits form here

  ros::Duration max_recording_time = ros::Duration(recordingTimeObject_);
  ros::Duration post_hit_recording_time = ros::Duration(recordingTimeRobot_);

  std::cout << "READY TO RECORD " << std::endl;

  while (ros::ok()) {

    // DEBUG
    if(print_count%20 == 0 ){
      // std::cout << "iiwa7_state : " << fsmState_.mode_iiwa7 << " \n iiwa14_state : " << fsmState_.mode_iiwa14<< std::endl;
      // std::cout << "time_since hit : " << time_since_hit << std::endl;
    }
    print_count +=1 ;

    // RECORD during hits
    if(isRecording_){
      
      // Update fixed values for data recording
      if(fsmState_.isHit){
        if(fsmState_.mode_iiwa7 == HIT){
          hittingTime_[IIWA_7] = fsmState_.hit_time;
          hittingFluxDes_[IIWA_7] = fsmState_.des_flux;
          desiredPosition_[IIWA_7] = fsmState_.des_pos;
        }
        else if(fsmState_.mode_iiwa14 == HIT){
          hittingTime_[IIWA_14] = fsmState_.hit_time;
          hittingFluxDes_[IIWA_14] = fsmState_.des_flux;
          desiredPosition_[IIWA_14] = fsmState_.des_pos;
        }
      }

      // Record data logic
      if(fsmState_.mode_iiwa7 == HIT){
        recordRobot(IIWA_7);
        recordObject(manual);
        write_once_7 = 1;
        write_once_object =1;
      }

      if(fsmState_.mode_iiwa14 == HIT){
        recordRobot(IIWA_14);
        recordObject(manual);
        write_once_14 = 1;
        write_once_object = 1;
      }

      //// OBJECT RECORDING LOGIC
      // Keep recording object for X seconds after hit (only when robots are at rest to avoid overlap)
      auto time_since_hit = ros::Time::now() - fsmState_.hit_time;
      if(fsmState_.mode_iiwa7 == REST && fsmState_.mode_iiwa14 == REST && time_since_hit <= max_recording_time && write_once_object){
        recordObject(manual);
      }
      // Stop recording object and write to file
      else if(fsmState_.mode_iiwa7 == REST && fsmState_.mode_iiwa14 == REST && 
              time_since_hit > max_recording_time && write_once_object){        
        std::string fn = recordingFolderPath_ + "object_hit_"+ std::to_string(hit_count)+".csv";
        writeObjectStatesToFile(hit_count, fn, manual);
        std::cout << "Finished writing hit " << hit_count << " for object!" << std::endl;
        hit_count += 1;
        write_once_object = 0;
        moved_manually_count_ = 1; // reset count for moved manually 
      }
      // If not during hit, check if we are moving the object manually, record if so
      else if(fsmState_.mode_iiwa7 == REST && fsmState_.mode_iiwa14 == REST && 
              time_since_hit > max_recording_time && !write_once_object){
        recordObjectMovedByHand(hit_count-1);
      }

      //// ROBOT RECORDING LOGIC 
      // keep recording robot data for X seconds after hit 
      if(fsmState_.mode_iiwa7 == REST && time_since_hit <= post_hit_recording_time && write_once_7){
        recordRobot(IIWA_7);
      }
      if(fsmState_.mode_iiwa14 == REST && time_since_hit <= post_hit_recording_time && write_once_14){
        recordRobot(IIWA_14);
      }

      // Writing data logic
      if(fsmState_.mode_iiwa7 == REST && time_since_hit > post_hit_recording_time && write_once_7){
        writeRobotStatesToFile(IIWA_7, hit_count);
        write_once_7 = 0;
      }

      if(fsmState_.mode_iiwa14 == REST && time_since_hit > post_hit_recording_time && write_once_14){
        writeRobotStatesToFile(IIWA_14, hit_count);
        write_once_14 = 0;
      }
    }

    ros::spinOnce();
    rate_.sleep();
  }

  ros::spinOnce();
  rate_.sleep();
  ros::shutdown();
}


int main(int argc, char** argv) {

  //ROS Initialization
  ros::init(argc, argv, "recorder");
  ros::NodeHandle nh;
  float frequency = 200.0f;

  std::unique_ptr<Recorder> record_hits = std::make_unique<Recorder>(nh, frequency);

  if (!record_hits->init()) {
    return -1;
  } else {
    std::cout << "OK "<< std::endl;;
    record_hits->run();
  }

  return 0;
}
