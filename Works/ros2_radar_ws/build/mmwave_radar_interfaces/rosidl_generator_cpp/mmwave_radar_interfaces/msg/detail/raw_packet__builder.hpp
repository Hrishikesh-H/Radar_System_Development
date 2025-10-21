// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mmwave_radar_interfaces:msg/RawPacket.idl
// generated code does not contain a copyright notice

#ifndef MMWAVE_RADAR_INTERFACES__MSG__DETAIL__RAW_PACKET__BUILDER_HPP_
#define MMWAVE_RADAR_INTERFACES__MSG__DETAIL__RAW_PACKET__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mmwave_radar_interfaces/msg/detail/raw_packet__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mmwave_radar_interfaces
{

namespace msg
{

namespace builder
{

class Init_RawPacket_data
{
public:
  Init_RawPacket_data()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::mmwave_radar_interfaces::msg::RawPacket data(::mmwave_radar_interfaces::msg::RawPacket::_data_type arg)
  {
    msg_.data = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mmwave_radar_interfaces::msg::RawPacket msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::mmwave_radar_interfaces::msg::RawPacket>()
{
  return mmwave_radar_interfaces::msg::builder::Init_RawPacket_data();
}

}  // namespace mmwave_radar_interfaces

#endif  // MMWAVE_RADAR_INTERFACES__MSG__DETAIL__RAW_PACKET__BUILDER_HPP_
