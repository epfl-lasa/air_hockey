//|
//|    Copyright (C) 2020 Learning Algorithms and Systems Laboratory, EPFL, Switzerland
//|    Authors:  Farshad Khadivr (maintainer)
//|    email:   farshad.khadivar@epfl.ch
//|    website: lasa.epfl.ch
//|
//|    This file is part of iiwa_toolkit.
//|
//|    iiwa_toolkit is free software: you can redistribute it and/or modify
//|    it under the terms of the GNU General Public License as published by
//|    the Free Software Foundation, either version 3 of the License, or
//|    (at your option) any later version.
//|
//|    iiwa_toolkit is distributed in the hope that it will be useful,
//|    but WITHOUT ANY WARRANTY; without even the implied warranty of
//|    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//|    GNU General Public License for more details.
//|

// #include <pinocchio/fwd.hpp>

#include <mutex>
#include <fstream>
#include <pthread.h>
#include <memory>
#include "std_msgs/Float64MultiArray.h"
#include "sensor_msgs/JointState.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/Inertia.h"

#include "ros/ros.h"
#include <ros/package.h>
#include <Eigen/Dense>

#include "passive_control.h"
#include "iiwa_toolkit/passive_cfg_paramsConfig.h"
#include "dynamic_reconfigure/server.h"

#define No_JOINTS 7
#define No_Robots 1
#define TOTAL_No_MARKERS 2



struct feedback
{
    Eigen::VectorXd jnt_position = Eigen::VectorXd(No_JOINTS);
    Eigen::VectorXd jnt_velocity = Eigen::VectorXd(No_JOINTS);
    Eigen::VectorXd jnt_torque = Eigen::VectorXd(No_JOINTS);
};


class IiwaRosMaster 
{
  public:
    IiwaRosMaster(ros::NodeHandle &n,double frequency):
    _n(n), _loopRate(frequency), _dt(1.0f/frequency){
        _stop =false;
    }

    ~IiwaRosMaster(){}

    bool init(){
        std::string ns = _n.getNamespace();
        std::string robot_name;
        if (ns == "/") {
            robot_name = "iiwa";
            ns = "/"+robot_name;
        } else if(ns.substr(0,1)=="/") {
            robot_name = ns.substr(1,ns.size()-1); 
        } else{
            robot_name = ns;
            ns = "/"+robot_name;
        }
        std::cout << "the namespace is: " + ns << " robot name is " << robot_name << std::endl;
        
        _feedback.jnt_position.setZero();
        _feedback.jnt_velocity.setZero();
        _feedback.jnt_torque.setZero();
        command_trq.setZero();
        command_plt.setZero();
        
        //!
        _subRobotStates[0]= _n.subscribe<sensor_msgs::JointState> (ns+"/joint_states", 1,
                boost::bind(&IiwaRosMaster::updateRobotStates,this,_1,0),ros::VoidPtr(),ros::TransportHints().reliable().tcpNoDelay());
        
        // _subOptitrack[0] = _n.subscribe<geometry_msgs::PoseStamped>("/vrpn_client_node/baseHand/pose", 1,
        //     boost::bind(&IiwaRosMaster::updateOptitrack,this,_1,0),ros::VoidPtr(),ros::TransportHints().reliable().tcpNoDelay());
        _subControl[0] = _n.subscribe<geometry_msgs::Pose>("/passive_control"+ns+"/pos_quat", 1,
            boost::bind(&IiwaRosMaster::updateControlPos,this,_1),ros::VoidPtr(),ros::TransportHints().reliable().tcpNoDelay());
        _subControl[1] = _n.subscribe<geometry_msgs::Pose>("/passive_control"+ns+"/vel_quat", 1,
            boost::bind(&IiwaRosMaster::updateControlVel,this,_1),ros::VoidPtr(),ros::TransportHints().reliable().tcpNoDelay());

        _TrqCmdPublisher = _n.advertise<std_msgs::Float64MultiArray>(ns+"/TorqueController/command",1);
        _JntPosCmdPublisher = _n.advertise<std_msgs::Float64MultiArray>(ns+"/PositionController/command",1);
        _EEPosePublisher = _n.advertise<geometry_msgs::Pose>(ns+"/ee_info/Pose",1);
        _EEVelPublisher = _n.advertise<geometry_msgs::Twist>(ns+"/ee_info/Vel",1);
        _InertiaPublisher = _n.advertise<geometry_msgs::Inertia>(ns+"/Inertia/taskPosInv", 1);
        _DirGradPublisher = _n.advertise<std_msgs::Float64MultiArray>(ns+"/Inertia/dirGrad",1);

        // Get the URDF XML from the parameter server
        std::string urdf_string, full_param;
        std::string robot_description = ns+"/robot_description";
        std::string end_effector;
        // gets the location of the robot description on the parameter server
        if (!_n.searchParam(robot_description, full_param)) {
            ROS_ERROR("Could not find parameter %s on parameter server", robot_description.c_str());
            return false;
        }
        // search and wait for robot_description on param server
        while (urdf_string.empty()) {
            ROS_INFO_ONCE_NAMED("Controller", "Controller is waiting for model"
                                                            " URDF in parameter [%s] on the ROS param server.",
                robot_description.c_str());
            _n.getParam(full_param, urdf_string);
            usleep(100000);
        }
        ROS_INFO_STREAM_NAMED("Controller", "Received urdf from param server, parsing...");

        // Get the end-effector
        _n.param<std::string>("params/end_effector", end_effector, robot_name+"_link_ee");

        // Initialize iiwa tools
        _controller = std::make_unique<PassiveControl>(urdf_string, end_effector);
        
        // Get Passive DS parameters
        if(!_n.getParam("control"+ns+"/dsGainPos", ds_gain_pos)){ROS_ERROR("Could not find Parameter dsGainPos");}
        if(!_n.getParam("control"+ns+"/dsGainOri", ds_gain_ori)){ROS_ERROR("Could not find Parameter dsGainOri");}
        if(!_n.getParam("control"+ns+"/lambda0Pos",lambda0_pos)){ROS_ERROR("Could not find Parameter lambda0Pos");}
        if(!_n.getParam("control"+ns+"/lambda1Pos",lambda1_pos)){ROS_ERROR("Could not find Parameter lambda1Pos");}
        if(!_n.getParam("control"+ns+"/lambda0Ori",lambda0_ori)){ROS_ERROR("Could not find Parameter lambda0Ori");}
        if(!_n.getParam("control"+ns+"/lambda1Ori",lambda1_ori)){ROS_ERROR("Could not find Parameter lambda1Ori");}
        if(!_n.getParam("control"+ns+"/alphaPos", alpha_pos)){ROS_ERROR("Could not find Parameter alphaPos");}
        if(!_n.getParam("control"+ns+"/alphaOri", alpha_ori)){ROS_ERROR("Could not find Parameter alphaOri");}
        if(!_n.getParam("control"+ns+"/lambda0PosHit",lambda0_hit)){ROS_ERROR("Could not find Parameter lambda0Pos");}
        _controller->set_pos_gains(ds_gain_pos,lambda0_pos,lambda1_pos, alpha_pos, lambda0_hit);
        _controller->set_ori_gains(ds_gain_ori,lambda0_ori,lambda1_ori,alpha_ori);

        // Get desired pose
        std::vector<double> dpos;
        std::vector<double> dquat;
        if(!_n.getParam("target"+ns+"/pos",dpos)){ROS_ERROR("Could not find Parameter target_pos");}
        if(!_n.getParam("target"+ns+"/quat",dquat)){ROS_ERROR("Could not find Parameter target_pos");}
        for (size_t i = 0; i < des_pos.size(); i++)
            des_pos(i) = dpos[i];
        for (size_t i = 0; i < des_quat.size(); i++)
            des_quat(i) = dquat[i]; 
        _controller->set_desired_pose(des_pos,des_quat);
        
        // Get inertia parameters
        std::vector<double> n_stiff;
        std::vector<double> n_damp;
        std::vector<double> hit_dir;
        std::vector<double> n_pos;
        if(!_n.getParam("inertia"+ns+"/null_stiffness", n_stiff)){ROS_ERROR("Could not find Parameter inertia null stiffness");}
        if(!_n.getParam("inertia"+ns+"/null_damping", n_damp)){ROS_ERROR("Could not find Parameter inertia null damping");}
        if(!_n.getParam("inertia"+ns+"/gain", inertia_gain)){ROS_ERROR("Could not find Parameter inertia gains");}
        if(!_n.getParam("inertia"+ns+"/desired", desired_inertia)){ROS_ERROR("Could not find Parameter desired inertia");}
        if(!_n.getParam("inertia"+ns+"/direction", hit_dir)){ROS_ERROR("Could not find Parameter inertia direction");}
        if(!_n.getParam("target"+ns+"/null_pos", n_pos)){ROS_ERROR("Could not find Parameter null_pos gains");}
        _controller->set_inertia_values(inertia_gain, desired_inertia);
        
        for (size_t i = 0; i < hit_direction.size(); i++)
            hit_direction(i) = hit_dir[i];
        _controller->set_hit_direction(hit_direction);

        for (size_t i = 0; i < null_stiff.size(); i++)
            null_stiff(i) = n_stiff[i];
        for (size_t i = 0; i < null_damp.size(); i++)
            null_damp(i) = n_damp[i];
        _controller->set_inertia_null_gains(null_stiff, null_damp);

        for (size_t i = 0; i < null_pos.size(); i++)
            null_pos(i) = n_pos[i];
        _controller->set_null_pos(null_pos);

        // Get PD gains for starting phase
        std::vector<double> stiffness;
        std::vector<double> damping;
        if(!_n.getParam("start"+ns+"/stiffness", stiffness)){ROS_ERROR("Could not find Parameter start stiffness gains");}
        if(!_n.getParam("start"+ns+"/damping", damping)){ROS_ERROR("Could not find Parameter start damping gains");}

        for (size_t i = 0; i < stiffness_gains.size(); i++)
            stiffness_gains(i) = stiffness[i];
        for (size_t i = 0; i < damping_gains.size(); i++)
            damping_gains(i) = damping[i];
        _controller->set_starting_phase_gains(stiffness_gains, damping_gains);

        // Get PD gains for Impedance orientation Control
        std::vector<double> stiff_ori;
        std::vector<double> damp_ori;
        if(!_n.getParam("control"+ns+"/ImpedanceOriStiffness", stiff_ori)){ROS_ERROR("Could not find Parameter Impedance Ori Stiffness gains");}
        if(!_n.getParam("control"+ns+"/ImpedanceOriDamping", damp_ori)){ROS_ERROR("Could not find Parameter Impedance Ori Damping gains");}

        for (size_t i = 0; i < stiffness_gains_ori.size(); i++)
            stiffness_gains_ori(i) = stiff_ori[i];
        for (size_t i = 0; i < damping_gains_ori.size(); i++)
            damping_gains_ori(i) = damp_ori[i];
        _controller->set_impedance_orientation_gains(stiffness_gains_ori, damping_gains_ori);

        // Get PD gains for Impedance position Control
        std::vector<double> stiff_pos;
        std::vector<double> damp_pos;
        if(!_n.getParam("control"+ns+"/ImpedancePosStiffness", stiff_pos)){ROS_ERROR("Could not find Parameter Impedance Pos Stifness gains");}
        if(!_n.getParam("control"+ns+"/ImpedancePosDamping", damp_pos)){ROS_ERROR("Could not find Parameter Impedance Pos Damping gains");}

        for (size_t i = 0; i < stiffness_gains_pos.size(); i++)
            stiffness_gains_pos(i) = stiff_pos[i];
        for (size_t i = 0; i < damping_gains_pos.size(); i++)
            damping_gains_pos(i) = damp_pos[i];
        _controller->set_impedance_position_gains(stiffness_gains_pos, damping_gains_pos);

        // plotting
        _plotPublisher = _n.advertise<std_msgs::Float64MultiArray>(ns+"/plotvar",1);
        
        // dynamic configure:
        if(!_n.getParam("use_rqt", use_rqt)){ROS_ERROR("Could not find Parameter use_rqt");}
        if(use_rqt){
            _dynRecCallback = boost::bind(&IiwaRosMaster::param_cfg_callback,this,_1,_2);
            _dynRecServer.setCallback(_dynRecCallback);
        }

        return true;
    }
    //
    // run node
    void run(){
        while(!_stop && ros::ok()){ 
            _mutex.lock();
                _controller->updateRobot(_feedback.jnt_position,_feedback.jnt_velocity,_feedback.jnt_torque);
                publishCommandTorque(_controller->getCmd());
                publishPlotVariable(command_plt);
                publishEEInfo();
                publishInertiaInfo();
                publishDirInertiaGrad();

                // publishPlotVariable(_controller->getPlotVariable());
                
            _mutex.unlock();
            
        ros::spinOnce();
        _loopRate.sleep();
        }
        publishCommandTorque(Eigen::VectorXd::Zero(No_JOINTS));
        ros::spinOnce();
        _loopRate.sleep();
        ros::shutdown();
    }

  protected:
    double _dt;

    ros::NodeHandle _n;
    ros::Rate _loopRate;

    ros::Subscriber _subRobotStates[No_Robots];
    ros::Subscriber _subControl[2];
    ros::Subscriber _subOptitrack[TOTAL_No_MARKERS];  // optitrack markers pose

    ros::Publisher _TrqCmdPublisher;
    ros::Publisher _JntPosCmdPublisher;
    ros::Publisher _EEPosePublisher;
    ros::Publisher _EEVelPublisher;
    ros::Publisher _InertiaPublisher;
    ros::Publisher _DirGradPublisher;

    ros::Publisher _plotPublisher;


    bool use_rqt;
    dynamic_reconfigure::Server<iiwa_toolkit::passive_cfg_paramsConfig> _dynRecServer;
    dynamic_reconfigure::Server<iiwa_toolkit::passive_cfg_paramsConfig>::CallbackType _dynRecCallback;


    feedback _feedback;
    // std::shared_ptr<iiwa_tools::IiwaTools> _tools;
    Eigen::VectorXd command_trq = Eigen::VectorXd(No_JOINTS);
    Eigen::VectorXd command_plt = Eigen::VectorXd(3);

    std::unique_ptr<PassiveControl> _controller;

    bool _stop;                        // Check for CTRL+C
    std::mutex _mutex;

    // std::unique_ptr<control::TrackControl> _controller;
    
    double ds_gain_pos;
    double ds_gain_ori;
    double lambda0_pos;
    double lambda1_pos;
    double lambda0_ori;
    double lambda1_ori;
    double alpha_pos;
    double alpha_ori;
    double lambda0_hit;
    Eigen::Vector3d des_pos = Eigen::Vector3d::Zero(); 
    Eigen::Vector4d des_quat = Eigen::Vector4d::Zero();

    Eigen::VectorXd null_stiff = Eigen::VectorXd::Zero(7);
    Eigen::VectorXd null_damp = Eigen::VectorXd::Zero(7);
    double inertia_gain;
    double desired_inertia;
    Eigen::Vector3d hit_direction = Eigen::Vector3d::Zero();
    Eigen::VectorXd null_pos = Eigen::VectorXd::Zero(7);

    // Start phase
    Eigen::VectorXd stiffness_gains = Eigen::VectorXd::Zero(7);
    Eigen::VectorXd damping_gains = Eigen::VectorXd::Zero(7);

    // Orientation Control
    Eigen::Vector3d stiffness_gains_ori = Eigen::Vector3d::Zero();
    Eigen::Vector3d damping_gains_ori = Eigen::Vector3d::Zero();

    // Position Control
    Eigen::Vector3d stiffness_gains_pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d damping_gains_pos = Eigen::Vector3d::Zero();

  private:

    void updateRobotStates(const sensor_msgs::JointState::ConstPtr &msg, int k){
       for (int i = 0; i < No_JOINTS; i++){
            _feedback.jnt_position[i] = (double)msg->position[i];
            _feedback.jnt_velocity[i] = (double)msg->velocity[i];
            _feedback.jnt_torque[i]   = (double)msg->effort[i];
        }

    }
    
    void updateTorqueCommand(const std_msgs::Float64MultiArray::ConstPtr &msg, int k){
       for (int i = 0; i < No_JOINTS; i++){
            command_trq[i] = (double)msg->data[i];
        }
    }
    
    void updatePlotVariable(const std_msgs::Float64MultiArray::ConstPtr &msg, int k){
       for (int i = 0; i < command_plt.size(); i++){
            command_plt[i] = (double)msg->data[i];
        }
    }
    // void updateBioTac(const biotac_sensors::BioTacHand &msg);
    void publishCommandTorque(const Eigen::VectorXd& cmdTrq){
        std_msgs::Float64MultiArray _cmd_jnt_torque;
        _cmd_jnt_torque.data.resize(No_JOINTS);

        if (cmdTrq.size() == No_JOINTS){
            for(int i = 0; i < No_JOINTS; i++)
                _cmd_jnt_torque.data[i] = cmdTrq[i];
            _TrqCmdPublisher.publish(_cmd_jnt_torque);
        }
    }
    void publishCommandJointPosition(const Eigen::VectorXd& cmdJntPos){
        std_msgs::Float64MultiArray _cmd_jnt_position;
        _cmd_jnt_position.data.resize(No_JOINTS);

        if (cmdJntPos.size() == No_JOINTS){
            for(int i = 0; i < No_JOINTS; i++)
                _cmd_jnt_position.data[i] = cmdJntPos[i];
            _JntPosCmdPublisher.publish(_cmd_jnt_position);
        }
    }
    void publishPlotVariable(const Eigen::VectorXd& pltVar){
        std_msgs::Float64MultiArray _plotVar;
        _plotVar.data.resize(pltVar.size());
        for (size_t i = 0; i < pltVar.size(); i++)
            _plotVar.data[i] = pltVar[i];
        _plotPublisher.publish(_plotVar);
    }
    void publishEEInfo(){
        geometry_msgs::Pose msg1;
        geometry_msgs::Twist msg2;

        Eigen::Vector3d pos = _controller->getEEpos();
        Eigen::Vector4d quat =  _controller->getEEquat();        
        Eigen::Vector3d vel =  _controller->getEEVel();
        Eigen::Vector3d angVel =  _controller->getEEAngVel();
        
        msg1.position.x  = pos[0];msg1.position.y  = pos[1];msg1.position.z  = pos[2];
        msg1.orientation.w = quat[0];msg1.orientation.x = quat[1];msg1.orientation.y = quat[2];msg1.orientation.z = quat[3];

        msg2.linear.x = vel[0];msg2.linear.y = vel[1];msg2.linear.z = vel[2];
        msg2.angular.x = angVel[0];msg2.angular.y = angVel[1];msg2.angular.z = angVel[2];

        _EEPosePublisher.publish(msg1);
        _EEVelPublisher.publish(msg2);
    }

    void publishInertiaInfo(){
        // Publishes task inertia in position only
        geometry_msgs::Inertia msg1;

        Eigen::MatrixXd task_inertia = _controller->getTaskInertiaPosInv();

        msg1.ixx = task_inertia(0, 0);
        msg1.ixy = task_inertia(0, 1);
        msg1.ixz = task_inertia(0, 2);
        msg1.iyy = task_inertia(1, 1);
        msg1.iyz = task_inertia(1, 2);
        msg1.izz = task_inertia(2, 2);

        _InertiaPublisher.publish(msg1);
    }
    void publishDirInertiaGrad(){
        std_msgs::Float64MultiArray _dir_grad;
        Eigen::VectorXd dirGrad = _controller->getDirTaskInertiaGrad();
        _dir_grad.data.resize(No_JOINTS);

        if (dirGrad.size() == No_JOINTS){
            for(int i = 0; i < No_JOINTS; i++)
                _dir_grad.data[i] = dirGrad[i];
            _DirGradPublisher.publish(_dir_grad);
        }
    }

    void updateControlPos(const geometry_msgs::Pose::ConstPtr& msg){
        Eigen::Vector3d pos;
        Eigen::Vector4d quat;

        pos << (double)msg->position.x, (double)msg->position.y, (double)msg->position.z;
        quat << (double)msg->orientation.w, (double)msg->orientation.x, (double)msg->orientation.y, (double)msg->orientation.z;
        
        if((pos.norm()>0)&&(pos.norm()<1.5)){
            _controller->set_desired_position(pos);
            if((quat.norm() >0)&&(quat.norm() < 1.1)){
                quat.normalize();
                _controller->set_desired_quat(quat);
            }
        }else{
            ROS_WARN("INCORRECT POSITIONING"); 
        }
    }

    void updateControlVel(const geometry_msgs::Pose::ConstPtr& msg){
        Eigen::Vector3d vel;
        Eigen::Vector4d quat;
        vel << (double)msg->position.x, (double)msg->position.y, (double)msg->position.z;
        quat << (double)msg->orientation.w, (double)msg->orientation.x, (double)msg->orientation.y, (double)msg->orientation.z;
        if(vel.norm()<5.0){
            _controller->set_desired_velocity(vel);
            if((quat.norm() > 0)&&(quat.norm() < 1.1)){
                quat.normalize();
                _controller->set_desired_quat(quat);
            }
        }else{
            ROS_WARN("VELOCITY OUT OF BOUND");
        }
    }

    void param_cfg_callback(iiwa_toolkit::passive_cfg_paramsConfig& config, uint32_t level){
        ROS_INFO("Reconfigure request.. Updating the parameters ... ");

        double sc_pos_ds = config.Position_DSgain;
        double sc_ori_ds = config.Orientation_DSgain;
        double sc_pos_lm = config.Position_lambda;
        double sc_ori_lm = config.Orientation_lambda;
        _controller->set_pos_gains(sc_pos_ds*ds_gain_pos,sc_pos_lm*lambda0_pos,sc_pos_lm*lambda1_pos, alpha_pos, lambda0_hit);
        _controller->set_ori_gains(sc_ori_ds*ds_gain_ori,sc_ori_lm*lambda0_ori,sc_ori_lm*lambda1_ori, alpha_ori);
        
        Eigen::Vector3d delta_pos = Eigen::Vector3d(config.dX_des,config.dY_des,config.dZ_des);

        // THIS DOES NOT WORK !!!!!
        // Eigen::Vector4d q_x = Eigen::Vector4d::Zero();
        // q_x(0) = std::cos(config.dX_des_angle/2);
        // q_x(1) = std::sin(config.dX_des_angle/2);
        // Eigen::Matrix3d rotMat_x = Utils<double>::quaternionToRotationMatrix(q_x);

        // Eigen::Vector4d q_y = Eigen::Vector4d::Zero();
        // q_y(0) = std::cos(config.dY_des_angle/2);
        // q_y(2) = std::sin(config.dY_des_angle/2);
        // Eigen::Matrix3d rotMat_y = Utils<double>::quaternionToRotationMatrix(q_y);

        // Eigen::Vector4d q_z = Eigen::Vector4d::Zero();
        // q_z(0) = std::cos(config.dZ_des_angle/2);
        // q_z(3) = std::sin(config.dZ_des_angle/2);
        // Eigen::Matrix3d rotMat_z = Utils<double>::quaternionToRotationMatrix(q_z);

        // //! this part has to be improved
        // Eigen::Matrix3d rotMat = rotMat_z*rotMat_y*rotMat_x * Utils<double>::quaternionToRotationMatrix(des_quat);

        _controller->set_desired_pose(des_pos+delta_pos,des_quat);

        double sc_inert_gain = config.Inertia_gain;
        double delta_inert_des = config.Inertia_desired;
        _controller->set_inertia_values(sc_inert_gain*inertia_gain, delta_inert_des+desired_inertia);
    }
};

//****************************************************
//****************************************************
int main (int argc, char **argv)
{
    float frequency = 200.0f;
    ros::init(argc,argv, "iiwa_passive_track");
    ros::NodeHandle n;

    std::unique_ptr<IiwaRosMaster> IiwaTrack = std::make_unique<IiwaRosMaster>(n,frequency);

    if (!IiwaTrack->init()){
        return -1;
    }else{
        IiwaTrack->run();
    }
    return 0;
}