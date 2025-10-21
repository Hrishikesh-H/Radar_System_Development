// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mmwave_radar_interfaces:msg/Heartbeat.idl
// generated code does not contain a copyright notice

#ifndef MMWAVE_RADAR_INTERFACES__MSG__DETAIL__HEARTBEAT__BUILDER_HPP_
#define MMWAVE_RADAR_INTERFACES__MSG__DETAIL__HEARTBEAT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mmwave_radar_interfaces/msg/detail/heartbeat__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mmwave_radar_interfaces
{

namespace msg
{

namespace builder
{

class Init_Heartbeat_frame_count
{
public:
  explicit Init_Heartbeat_frame_count(::mmwave_radar_interfaces::msg::Heartbeat & msg)
  : msg_(msg)
  {}
  ::mmwave_radar_interfaces::msg::Heartbeat frame_count(::mmwave_radar_interfaces::msg::Heartbeat::_frame_count_type arg)
  {
    msg_.frame_count = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mmwave_radar_interfaces::msg::Heartbeat msg_;
};

class Init_Heartbeat_builtin_time
{
public:
  Init_Heartbeat_builtin_time()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Heartbeat_frame_count builtin_time(::mmwave_radar_interfaces::msg::Heartbeat::_builtin_time_type arg)
  {
    msg_.builtin_time = std::move(arg);
    return Init_Heartbeat_frame_count(msg_);
  }

private:
  ::mmwave_radar_interfaces::msg::Heartbeat msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::mmwave_radar_interfaces::msg::Heartbeat>()
{
  return mmwave_radar_interfaces::msg::builder::Init_Heartbeat_builtin_time();
}

}  // namespace mmwave_radar_interfaces

#endif  // MMWAVE_RADAR_INTERFACES__MSG__DETAIL__HEARTBEAT__BUILDER_HPP_
