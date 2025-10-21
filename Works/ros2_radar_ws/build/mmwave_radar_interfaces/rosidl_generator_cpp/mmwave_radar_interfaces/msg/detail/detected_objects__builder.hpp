// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mmwave_radar_interfaces:msg/DetectedObjects.idl
// generated code does not contain a copyright notice

#ifndef MMWAVE_RADAR_INTERFACES__MSG__DETAIL__DETECTED_OBJECTS__BUILDER_HPP_
#define MMWAVE_RADAR_INTERFACES__MSG__DETAIL__DETECTED_OBJECTS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mmwave_radar_interfaces/msg/detail/detected_objects__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mmwave_radar_interfaces
{

namespace msg
{

namespace builder
{

class Init_DetectedObjects_velocity
{
public:
  explicit Init_DetectedObjects_velocity(::mmwave_radar_interfaces::msg::DetectedObjects & msg)
  : msg_(msg)
  {}
  ::mmwave_radar_interfaces::msg::DetectedObjects velocity(::mmwave_radar_interfaces::msg::DetectedObjects::_velocity_type arg)
  {
    msg_.velocity = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mmwave_radar_interfaces::msg::DetectedObjects msg_;
};

class Init_DetectedObjects_z
{
public:
  explicit Init_DetectedObjects_z(::mmwave_radar_interfaces::msg::DetectedObjects & msg)
  : msg_(msg)
  {}
  Init_DetectedObjects_velocity z(::mmwave_radar_interfaces::msg::DetectedObjects::_z_type arg)
  {
    msg_.z = std::move(arg);
    return Init_DetectedObjects_velocity(msg_);
  }

private:
  ::mmwave_radar_interfaces::msg::DetectedObjects msg_;
};

class Init_DetectedObjects_y
{
public:
  explicit Init_DetectedObjects_y(::mmwave_radar_interfaces::msg::DetectedObjects & msg)
  : msg_(msg)
  {}
  Init_DetectedObjects_z y(::mmwave_radar_interfaces::msg::DetectedObjects::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_DetectedObjects_z(msg_);
  }

private:
  ::mmwave_radar_interfaces::msg::DetectedObjects msg_;
};

class Init_DetectedObjects_x
{
public:
  explicit Init_DetectedObjects_x(::mmwave_radar_interfaces::msg::DetectedObjects & msg)
  : msg_(msg)
  {}
  Init_DetectedObjects_y x(::mmwave_radar_interfaces::msg::DetectedObjects::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_DetectedObjects_y(msg_);
  }

private:
  ::mmwave_radar_interfaces::msg::DetectedObjects msg_;
};

class Init_DetectedObjects_num_obj
{
public:
  Init_DetectedObjects_num_obj()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_DetectedObjects_x num_obj(::mmwave_radar_interfaces::msg::DetectedObjects::_num_obj_type arg)
  {
    msg_.num_obj = std::move(arg);
    return Init_DetectedObjects_x(msg_);
  }

private:
  ::mmwave_radar_interfaces::msg::DetectedObjects msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::mmwave_radar_interfaces::msg::DetectedObjects>()
{
  return mmwave_radar_interfaces::msg::builder::Init_DetectedObjects_num_obj();
}

}  // namespace mmwave_radar_interfaces

#endif  // MMWAVE_RADAR_INTERFACES__MSG__DETAIL__DETECTED_OBJECTS__BUILDER_HPP_
