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

#include "passive_control.h"

PassiveDS::PassiveDS(const double& lam0, const double& lam1):eigVal0(lam0),eigVal1(lam1){
    set_damping_eigval(lam0,lam1);
}

PassiveDS::~PassiveDS(){}
void PassiveDS::set_damping_eigval(const double& lam0, const double& lam1){
    if((lam0 > 0)&&(lam1 > 0)){
        eigVal0 = lam0;
        eigVal1 = lam1;
        damping_eigval(0,0) = eigVal0;
        damping_eigval(1,1) = eigVal1;
        damping_eigval(2,2) = eigVal1;
    }else{
        std::cerr << "wrong values for the eigenvalues"<<"\n";
    }
}
void PassiveDS::updateDampingMatrix(const Eigen::Vector3d& ref_vel){ 

    if(ref_vel.norm() > 1e-6){
        baseMat.setRandom();
        baseMat.col(0) = ref_vel.normalized();
        for(uint i=1;i<3;i++){
            for(uint j=0;j<i;j++)
                baseMat.col(i) -= baseMat.col(j).dot(baseMat.col(i))*baseMat.col(j);
            baseMat.col(i).normalize();
        }
        Dmat = baseMat*damping_eigval*baseMat.transpose();
    }else{
        Dmat = Eigen::Matrix3d::Identity();
    }
    // otherwise just use the last computed basis
}

void PassiveDS::update(const Eigen::Vector3d& vel, const Eigen::Vector3d& des_vel){
    // compute damping
    updateDampingMatrix(des_vel);
    // dissipate
    control_output = - Dmat * vel;
    // compute control
    control_output += eigVal0*des_vel;
}
Eigen::Vector3d PassiveDS::get_output(){ return control_output;}

//************************************************

PassiveControl::PassiveControl(const std::string& urdf_string,const std::string& end_effector)
{
    _tools.init_rbdyn(urdf_string, end_effector);

    dsGain_pos = 5.00;
    dsGain_ori = 2.50;

    dsContPos = std::make_unique<PassiveDS>( 100., 100.);
    dsContOri = std::make_unique<PassiveDS>(5., 5.);
    
  
    _robot.name +=std::to_string(0);
    _robot.jnt_position.setZero();
    _robot.jnt_velocity.setZero();
    _robot.jnt_torque.setZero();
    _robot.nulljnt_position.setZero();
    _robot.ee_pos.setZero(); 
    _robot.ee_vel.setZero();   
    _robot.ee_acc.setZero();

    
    double angle0 = 0.25*M_PI;
    _robot.ee_quat[0] = (std::cos(angle0/2));
    _robot.ee_quat.segment(1,3) = (std::sin(angle0/2))* Eigen::Vector3d::UnitZ();
    
    _robot.ee_angVel.setZero();
    _robot.ee_angAcc.setZero();
   
   //* desired things
    _robot.ee_des_pos = {-0.5 , 0.5, 0.2}; 
    double angled = 1.0*M_PI;
    _robot.ee_des_quat[0] = (std::cos(angled/2));
    _robot.ee_des_quat.segment(1,3) = (std::sin(angled/2))* Eigen::Vector3d::UnitX();
    
    //** do we need these parts in here?
    _robot.ee_des_vel.setZero();   
    _robot.ee_des_acc.setZero();
    _robot.ee_des_angVel.setZero();
    _robot.ee_des_angAcc.setZero();


    _robot.jacob.setZero();
    _robot.jacob.setZero();       
    _robot.jacob_drv.setZero();   
    _robot.jacob_t_pinv.setZero();
    _robot.jacobPos.setZero();   
    _robot.jacobAng.setZero();
    _robot.pseudo_inv_jacob.setZero();   
    _robot.pseudo_inv_jacobPos.setZero();

    // _robot.nulljnt_position << 0.0, 0.0, 0.0, -.75, 0., 0.0, 0.0;
    // _robot.nulljnt_position << -1.09, 1.59, 1.45, -1.6, -2.81, 0.0, 0.0; // inertia
    // _robot.nulljnt_position << 0.00, -0.6, 0.50, -1.7, 0.00, 0.00, 0.0; // manipulability  0.00, -0.6, 0.00, -1.7, 0.00, 0.00, 0.0;
    // _robot.nulljnt_position << 0.549, 1.110, -0.121, -1.382, -1.088, 1.816, 1.041; // 
}

PassiveControl::~PassiveControl(){}



void PassiveControl::updateRobot(const Eigen::VectorXd& jnt_p,const Eigen::VectorXd& jnt_v,const Eigen::VectorXd& jnt_t){
    

    _robot.jnt_position = jnt_p;
    _robot.jnt_velocity = jnt_v;
    _robot.jnt_torque   = jnt_t;

    iiwa_tools::RobotState robot_state;
    
    robot_state.position.resize(jnt_p.size());
    robot_state.velocity.resize(jnt_p.size());
    for (size_t i = 0; i < jnt_p.size(); i++) {
        robot_state.position[i] = _robot.jnt_position[i];
        robot_state.velocity[i] = _robot.jnt_velocity[i];
    }

    std::tie(_robot.jacob, _robot.jacob_drv) = _tools.jacobians(robot_state);
    _robot.jacobPos =  _robot.jacob.bottomRows(3);
    _robot.jacobAng =  _robot.jacob.topRows(3);

    _robot.pseudo_inv_jacob    = pseudo_inverse(Eigen::MatrixXd(_robot.jacob * _robot.jacob.transpose()) );
    _robot.pseudo_inv_jacobPos = pseudo_inverse(Eigen::MatrixXd(_robot.jacobPos * _robot.jacobPos.transpose()) );
    // _robot.pseudo_inv_jacobPJnt = pseudo_inverse(Eigen::MatrixXd(_robot.jacobPos.transpose() * _robot.jacobPos ) );
    _robot.pseudo_inv_jacobJnt = pseudo_inverse(Eigen::MatrixXd(_robot.jacob.transpose() * _robot.jacob ) );
    

    _robot.joint_inertia = _tools.get_joint_inertia(robot_state);
    _robot.task_inertiaPos = jointToTaskInertia(_robot.jacobPos, _robot.joint_inertia);
    _robot.task_inertiaPos_inv = jointToTaskInertiaInverse(_robot.jacobPos, _robot.joint_inertia);
    _robot.dir_task_inertia_grad = getDirInertiaGrad(robot_state, _robot.direction);
    
    auto ee_state = _tools.perform_fk(robot_state);
    _robot.ee_pos = ee_state.translation;
    _robot.ee_quat[0] = ee_state.orientation.w();
    _robot.ee_quat.segment(1,3) = ee_state.orientation.vec();
    

    Eigen::VectorXd vel = _robot.jacob * _robot.jnt_velocity;
    _robot.ee_vel    = vel.tail(3); // check whether this is better or filtering position derivitive

    _robot.ee_angVel = vel.head(3); // compare it with your quaternion derivitive equation

    // for(int i = 0; i < 3; i++)
    //     _plotVar.data[i] = (_robot.ee_des_vel - _robot.ee_vel)[i];
}

Eigen::Vector3d PassiveControl::getEEpos(){
    return _robot.ee_pos;
}

Eigen::Vector4d PassiveControl::getEEquat(){
    return _robot.ee_quat;
}
Eigen::Vector3d PassiveControl::getEEVel(){
    return _robot.ee_vel;
}
Eigen::Vector3d PassiveControl::getEEAngVel(){
    return _robot.ee_angVel;
}

Eigen::MatrixXd PassiveControl::jointToTaskInertia(const Eigen::MatrixXd& Jac, const Eigen::MatrixXd& joint_inertia){
    Eigen::MatrixXd task_inertia_inverse = Jac * joint_inertia.inverse() * Jac.transpose();
    Eigen::MatrixXd task_inertia = task_inertia_inverse.inverse();
    return task_inertia;
}

Eigen::MatrixXd PassiveControl::jointToTaskInertiaInverse(const Eigen::MatrixXd& Jac, const Eigen::MatrixXd& joint_inertia){
    Eigen::MatrixXd task_inertia_inverse = Jac * joint_inertia.inverse() * Jac.transpose();
    // Eigen::MatrixXd task_inertia = task_inertia_inverse.inverse();
    return task_inertia_inverse;
}

Eigen::MatrixXd PassiveControl::getTaskInertiaPosInv(){
    return _robot.task_inertiaPos_inv;
}

Eigen::VectorXd PassiveControl::getDirInertiaGrad(iiwa_tools::RobotState &current_state, Eigen::Vector3d &direction){
    // Eigen::MatrixXd task_inertia = _robot.task_inertiaPos;
    // double dir_inertia = direction.transpose() * task_inertia * direction;
    double dir_inertia = 1/(direction.transpose() * _robot.task_inertiaPos_inv * direction);
    Eigen::VectorXd grad = Eigen::VectorXd(7);
    Eigen::MatrixXd duplicate_joint_inertia = Eigen::MatrixXd(7,7);
    Eigen::MatrixXd duplicate_task_inertiaPos = Eigen::MatrixXd(6,6);
    Eigen::MatrixXd duplicate_task_inertiaPos_inverse = Eigen::MatrixXd(6,6);
    Eigen::MatrixXd jacob = Eigen::MatrixXd(6, 7);
    Eigen::MatrixXd jacob_drv = Eigen::MatrixXd(6, 7);
    Eigen::MatrixXd jacobPos = Eigen::MatrixXd(3, 7);
    double duplicate_dir_inertia;

    // create a duplicate current state
    iiwa_tools::RobotState duplicate_state = current_state;
    double dq = 0.001;
    for(int i = 0; i < 7; ++i){
        duplicate_state.position[i]+=dq;
        duplicate_joint_inertia = _tools.get_joint_inertia(duplicate_state);

        std::tie(jacob, jacob_drv) = _tools.jacobians(duplicate_state);
        jacobPos = jacob.bottomRows(3);
        // duplicate_task_inertiaPos = jointToTaskInertia(jacobPos, duplicate_joint_inertia);
        duplicate_task_inertiaPos_inverse = jointToTaskInertiaInverse(jacobPos, duplicate_joint_inertia);
        duplicate_dir_inertia = 1/(direction.transpose() * duplicate_task_inertiaPos_inverse * direction);

        grad[i] = (duplicate_dir_inertia - dir_inertia) / dq;

        duplicate_state.position[i]-=dq;
    }

    return grad;
}


void PassiveControl::set_pos_gains(const double& ds, const double& lambda0,const double& lambda1){
    dsGain_pos = ds;
    dsContPos->set_damping_eigval(lambda0,lambda1);
}

void PassiveControl::set_ori_gains(const double& ds, const double& lambda0,const double& lambda1){
    dsGain_ori = ds;
    dsContOri->set_damping_eigval(lambda0,lambda1);
}

void PassiveControl::set_null_pos(const Eigen::VectorXd& nullPosition){
    if (nullPosition.size() == _robot.nulljnt_position.size() )
    {
        _robot.nulljnt_position = nullPosition;
    }else{
        ROS_ERROR("wrong size for the null joint position");
    }
}

void PassiveControl::set_desired_pose(const Eigen::Vector3d& pos, const Eigen::Vector4d& quat){
    _robot.ee_des_pos = pos;
    _robot.ee_des_quat = quat;
    is_just_velocity = false;
}
void PassiveControl::set_desired_position(const Eigen::Vector3d& pos){
    _robot.ee_des_pos = pos;
     is_just_velocity = false;
}
void PassiveControl::set_desired_quat(const Eigen::Vector4d& quat){
    _robot.ee_des_quat = quat;
}
void PassiveControl::set_desired_velocity(const Eigen::Vector3d& vel){
     _robot.ee_des_vel = vel;
     is_just_velocity = true;
}

void PassiveControl::set_load(const double& mass ){
    load_added = mass;
}

void PassiveControl::set_inertia_null_gains(const Eigen::VectorXd& null_mat){
    null_gains = null_mat;
}

void PassiveControl::set_hit_direction(const Eigen::Vector3d& direction){
    _robot.direction = direction;
}

void PassiveControl::set_inertia_values(const double& gain, const double& desired){
    inertia_gain = gain;
    desired_inertia = desired;
}  

void PassiveControl::set_starting_phase_gains(const Eigen::VectorXd& stiff_gains, const Eigen::VectorXd& damp_gains){
    start_stiffness_gains = stiff_gains;
    start_damping_gains = damp_gains;
}

void PassiveControl::set_impedance_orientation_gains(const Eigen::VectorXd& stiff_gains, const Eigen::VectorXd& damp_gains){
    K_r(0,0) = stiff_gains[0]; 
    K_r(1,1) = stiff_gains[1]; 
    K_r(2,2) = stiff_gains[2];  

    B_ang(0,0) = damp_gains[0];
    B_ang(1,1) = damp_gains[1];
    B_ang(2,2) = damp_gains[2];
}

void PassiveControl::set_impedance_position_gains(const Eigen::VectorXd& stiff_gains, const Eigen::VectorXd& damp_gains){
    K_t(0,0) = stiff_gains[0]; 
    K_t(1,1) = stiff_gains[1]; 
    K_t(2,2) = stiff_gains[2];  

    B_lin(0,0) = damp_gains[0];
    B_lin(1,1) = damp_gains[1];
    B_lin(2,2) = damp_gains[2];
}

Eigen::VectorXd PassiveControl::computeInertiaTorqueNull(float des_dir_lambda, Eigen::Vector3d& des_vel){
    
    // Eigen::Vector3d direction = des_vel / des_vel.norm();
    float inertia_error = 1/(_robot.direction.transpose() * _robot.task_inertiaPos_inv * _robot.direction) - des_dir_lambda;
    Eigen::VectorXd null_torque = 1.0 * _robot.dir_task_inertia_grad * inertia_error;
    // Eigen::VectorXd null_torque = 1.0 * _robot.dir_task_inertia_grad ;

    return null_torque;
}

void PassiveControl::computeTorqueCmd(){

    //// POSITION CONTROL
    bool pos_ctrl_ds = true;
    Eigen::VectorXd tmp_jnt_trq_pos = Eigen::VectorXd::Zero(7);

    if(pos_ctrl_ds){
        // desired position values
        Eigen::Vector3d deltaX = _robot.ee_des_pos - _robot.ee_pos;
        double maxDx = 0.1;
        if (deltaX.norm() > maxDx)
            deltaX = maxDx * deltaX.normalized();
        
        double theta_g = (-.5/(4*maxDx*maxDx)) * deltaX.transpose() * deltaX;
        
        Eigen::Matrix3d zgain = Eigen::Matrix3d::Identity();
        zgain(0,0) *= 1.5; 
        zgain(2,2) *= 1.5; 

        Eigen::Matrix3d xgain = Eigen::Matrix3d::Identity();
        xgain(0,0) *= 1.5; 

        if(!is_just_velocity)
            _robot.ee_des_vel   = zgain * dsGain_pos*(Eigen::Matrix3d::Identity()+xgain*std::exp(theta_g)) *deltaX;
        
        // -----------------------get desired force in task space
        dsContPos->update(_robot.ee_vel,_robot.ee_des_vel);
        Eigen::Vector3d wrenchPos = dsContPos->get_output() + load_added * 9.8*Eigen::Vector3d::UnitZ();   
        tmp_jnt_trq_pos = _robot.jacobPos.transpose() * wrenchPos;
    }
    else{
        // Impedance control
        // Adapt to receiving des_vel or des_pos
        if(is_just_velocity){ // getting onyl vel -> set desired pos to be next step
            _robot.ee_des_pos =  _robot.ee_pos + _robot.ee_des_vel * 0.005;
        }
        else if(!is_just_velocity){ // getting onyl pos -> reset des vel
            _robot.ee_des_vel = Eigen::Vector3d::Zero();
        }

        // Torque of lin. wrench
        Eigen::VectorXd w_0_trans  = -K_t * ( _robot.ee_pos - _robot.ee_des_pos); 
        Eigen::VectorXd tau_elastic_linK = _robot.jacobPos.transpose() * w_0_trans;
         
        // Torque of lin. vel
        Eigen::Vector3d w_lin_0_damp = B_lin * (_robot.ee_des_vel - _robot.ee_vel);
        Eigen::VectorXd tau_damp_lin = _robot.jacobPos.transpose() * w_lin_0_damp;

        tmp_jnt_trq_pos = tau_elastic_linK + tau_damp_lin;
    }


    // ORIENTATION CONTROL
    bool ori_ctrl_ds = false;
    Eigen::VectorXd tmp_jnt_trq_ang = Eigen::VectorXd::Zero(7);

    if(ori_ctrl_ds){
        // desired angular values
        Eigen::Vector4d dqd = Utils<double>::slerpQuaternion(_robot.ee_quat, _robot.ee_des_quat, 0.5);    
        Eigen::Vector4d deltaQ = dqd -  _robot.ee_quat;

        Eigen::Vector4d qconj = _robot.ee_quat;
        qconj.segment(1,3) = -1 * qconj.segment(1,3);
        Eigen::Vector4d temp_angVel = Utils<double>::quaternionProduct(deltaQ, qconj);

        Eigen::Vector3d tmp_angular_vel = temp_angVel.segment(1,3);
        double maxDq = 0.2;
        if (tmp_angular_vel.norm() > maxDq)
            tmp_angular_vel = maxDq * tmp_angular_vel.normalized();

        double theta_gq = (-.5/(4*maxDq*maxDq)) * tmp_angular_vel.transpose() * tmp_angular_vel;
        _robot.ee_des_angVel  = 2 * dsGain_ori*(1+std::exp(theta_gq)) * tmp_angular_vel;

        // Orientation
        dsContOri->update(_robot.ee_angVel,_robot.ee_des_angVel);
        Eigen::Vector3d wrenchAng = dsContOri->get_output();
        tmp_jnt_trq_ang = _robot.jacobAng.transpose() * wrenchAng;
        
    }
    else{
        // Impedance control for orientation torque
        Eigen::Matrix3d R_tcp_des = Eigen::Matrix3d::Zero(3,3);
        Eigen::Matrix3d R_0_d = Eigen::Matrix3d::Zero(3,3);
        Eigen::Matrix3d R_0_tcp = Eigen::Matrix3d::Zero(3,3);
        Eigen::Vector3d u_p = Eigen::Vector3d::Zero(); 

        R_0_d = Utils<double>::quaternionToRotationMatrix(_robot.ee_des_quat);
        R_0_tcp = Utils<double>::quaternionToRotationMatrix(_robot.ee_quat);
        R_tcp_des = R_0_tcp.transpose() * R_0_d; // R_0_d desired quaternion, R_0_tcp -> ee_quat
        Eigen::Quaterniond Qdes(R_tcp_des);
        Qdes.normalize();
        double fact = 0;
        double theta_des = 2 * acos(Qdes.w());
        if(theta_des == 0.0){
            fact = 0;
        } else{
            fact = 1 / (sin(theta_des/2));
        }
        u_p(0) = fact * Qdes.x();
        u_p(1) = fact * Qdes.y();
        u_p(2) = fact * Qdes.z();
        
        Eigen::Vector3d u_0 = R_0_tcp * u_p;

        Eigen::Vector3d wrenchAng = K_r * u_0 * theta_des;

        Eigen::VectorXd tau_elastic_rotK = _robot.jacobAng.transpose() * wrenchAng;

        // Add Damping
        // Desired angular speed is ZERO
        Eigen::Vector3d v_ang_0_des = Eigen::Vector3d::Zero();

        // Torque for ang. vel 
        Eigen::Vector3d w_0_damp_ang = B_ang * (v_ang_0_des - _robot.ee_angVel);

        Eigen::VectorXd tau_damp_ang =  _robot.jacobAng.transpose() * w_0_damp_ang;
        
        tmp_jnt_trq_ang = tau_elastic_rotK + tau_damp_ang;

        std::cout << "Angular torque is: " << tmp_jnt_trq_ang << std::endl;
        // std::cout << "Angular des quat is: " << _robot.ee_des_quat << std::endl;
    }

    //sum up:
    Eigen::VectorXd tmp_jnt_trq = tmp_jnt_trq_pos + tmp_jnt_trq_ang;

    // null pos control
    Eigen::MatrixXd tempMat2 =  Eigen::MatrixXd::Identity(7,7) - _robot.jacob.transpose()* _robot.pseudo_inv_jacob* _robot.jacob;
    Eigen::VectorXd er_null;

    // Use different nullspace depending on whether we are going to a position or tracking a velocity
    if(!is_just_velocity){
        er_null = _robot.jnt_position -_robot.nulljnt_position;
    }
    else{
        er_null = inertia_gain*computeInertiaTorqueNull(desired_inertia, _robot.ee_des_vel); 
    }

    // PD control when starting to reach null space joint position
    Eigen::VectorXd tmp_null_trq = Eigen::VectorXd::Zero(7);

    if(first){ // PD torque controller with big damping for initialization 
        // compute PD torque 
        for (int i =0; i<7; i++){ 
             tmp_null_trq[i] = -(start_stiffness_gains[i] * er_null[i]); // Stiffness
            tmp_null_trq[i] += -start_damping_gains[i] * _robot.jnt_velocity[i]; // Damping            
        }
        // std::cout << "PD torque is: " << tmp_null_trq << std::endl;
        ROS_INFO_ONCE("Using PD torque control  "); 
        
        if(er_null.norm()<0.15){ // Go back to usual control when close enough 
            first = false;
            ROS_INFO_ONCE("Close to first pose. Stopping PD control."); 
        } 

        // PD joint control 
        _trq_cmd = tmp_null_trq;
    }
    else{ // USUAL CONTROL
        ROS_INFO_ONCE("DS control activated");

        // compute null torque
        if (er_null.norm()>2e-1){
            er_null = 0.2*er_null.normalized();
        }

        for (int i =0; i<7; i++){ 
            tmp_null_trq[i] = -null_gains[i] * er_null[i];
            tmp_null_trq[i] +=-1. * _robot.jnt_velocity[i];
        }
        tmp_null_trq = 10*tempMat2 * tmp_null_trq;
   
        _trq_cmd = tmp_jnt_trq + tmp_null_trq;
    }
   

    // Gravity Compensationn
    // the gravity compensation should've been here, but a server form iiwa tools is doing the job.
   
}
