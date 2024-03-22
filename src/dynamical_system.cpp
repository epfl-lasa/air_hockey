//|    Copyright (C) 2020 Learning Algorithms and Systems Laboratory, EPFL, Switzerland
//|    Authors:  Harshit Khurana (maintainer)
//|    email:   harshit.khurana@epfl.ch
//|    website: lasa.epfl.ch

#include "dynamical_system.h"

hitting_DS::hitting_DS(Eigen::Vector3f& current_end_effector, Eigen::Vector3f& attractor_main) {
  current_position_ = current_end_effector;
  DS_attractor_ = attractor_main;
}
hitting_DS::hitting_DS(Eigen::Vector3f& current_end_effector) { current_position_ = current_end_effector; }
hitting_DS::hitting_DS(Eigen::Vector3f& current_end_effector,
                       Eigen::Vector3f& desired_position,
                       Eigen::Vector3f& hit_direction,
                       float& hit_speed) {
  current_position_ = current_end_effector;
  DS_attractor_ = desired_position;
  des_direction_ = hit_direction;
  des_speed_ = hit_speed;
}

Eigen::Vector3f hitting_DS::flux_DS(float dir_flux, Eigen::Matrix3f& current_inertia_inverse) {

  /* ** Finding the virtual end effector position ** */

  Eigen::Vector3f reference_velocity = Eigen::Vector3f{0.0, 0.0, 0.0};
  // Eigen::Vector3f reference_direction = Eigen::Vector3f{0.0, 1.0, 0.0};
  Eigen::Vector3f relative_position = current_position_ - DS_attractor_;
  Eigen::Vector3f virtual_ee =
      DS_attractor_ + des_direction_ * (relative_position.dot(des_direction_) / (des_direction_.squaredNorm()));

  float dir_inertia = 1/(des_direction_.transpose() * current_inertia_inverse * des_direction_);

  // ROS_INFO_STREAM("Dir_inertia: " << dir_inertia);
  float exp_term = (current_position_ - virtual_ee).norm();
  float alpha = exp(-exp_term / (sigma_ * sigma_));

  reference_velocity = alpha * des_direction_ + (1 - alpha) * gain_ * (current_position_ - virtual_ee);

  reference_velocity =
      (dir_flux / dir_inertia) * (dir_inertia + m_obj_) * reference_velocity / reference_velocity.norm();

  return reference_velocity;
}
Eigen::Vector3f hitting_DS::linear_DS() { return gain_ * (current_position_ - DS_attractor_); }
Eigen::Vector3f hitting_DS::linear_DS(Eigen::Vector3f& attractor) { return gain_ * (current_position_ - attractor); }
Eigen::Vector3f hitting_DS::vel_max_DS() {
  Eigen::Vector3f reference_velocity = Eigen::Vector3f{0.0, 0.0, 0.0};
  Eigen::Vector3f relative_position = current_position_ - DS_attractor_;
  Eigen::Vector3f virtual_ee =
      DS_attractor_ + des_direction_ * (relative_position.dot(des_direction_) / (des_direction_.squaredNorm()));
  float exp_term = (current_position_ - virtual_ee).norm();
  float alpha = exp(-exp_term / (sigma_ * sigma_));

  reference_velocity = alpha * des_direction_ + (1 - alpha) * gain_ * (current_position_ - virtual_ee);
  reference_velocity = des_speed_ * reference_velocity / reference_velocity.norm();

  //ROS_INFO_STREAM("ref: " << reference_velocity.transpose());

  return reference_velocity;
}

Eigen::Vector3f hitting_DS::get_des_direction() { return des_direction_; }
Eigen::Vector3f hitting_DS::get_current_position() { return current_position_; }
Eigen::Vector3f hitting_DS::get_DS_attractor() { return DS_attractor_; }
Eigen::Matrix3f hitting_DS::get_gain() { return gain_; }

void hitting_DS::set_des_direction(Eigen::Vector3f desired_direction) { des_direction_ = desired_direction; }
void hitting_DS::set_current_position(Eigen::Vector3f current_position) { current_position_ = current_position; }
void hitting_DS::set_DS_attractor(Eigen::Vector3f DS_attractor) { DS_attractor_ = DS_attractor; }
void hitting_DS::set_gain(Eigen::Matrix3f gain) { gain_ = gain; }
