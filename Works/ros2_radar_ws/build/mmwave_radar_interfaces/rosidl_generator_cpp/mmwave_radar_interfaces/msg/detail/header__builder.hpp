// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mmwave_radar_interfaces:msg/Header.idl
// generated code does not contain a copyright notice

#ifndef MMWAVE_RADAR_INTERFACES__MSG__DETAIL__HEADER__BUILDER_HPP_
#define MMWAVE_RADAR_INTERFACES__MSG__DETAIL__HEADER__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mmwave_radar_interfaces/msg/detail/header__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mmwave_radar_interfaces
{

namespace msg
{

namespace builder
{

class Init_Header_subframe_number
{
public:
  explicit Init_Header_subframe_number(::mmwave_radar_interfaces::msg::Header & msg)
  : msg_(msg)
  {}
  ::mmwave_radar_interfaces::msg::Header subframe_number(::mmwave_radar_interfaces::msg::Header::_subframe_number_type arg)
  {
    msg_.subframe_number = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mmwave_radar_interfaces::msg::Header msg_;
};

class Init_Header_num_tlvs
{
public:
  explicit Init_Header_num_tlvs(::mmwave_radar_interfaces::msg::Header & msg)
  : msg_(msg)
  {}
  Init_Header_subframe_number num_tlvs(::mmwave_radar_interfaces::msg::Header::_num_tlvs_type arg)
  {
    msg_.num_tlvs = std::move(arg);
    return Init_Header_subframe_number(msg_);
  }

private:
  ::mmwave_radar_interfaces::msg::Header msg_;
};

class Init_Header_num_detected_obj
{
public:
  explicit Init_Header_num_detected_obj(::mmwave_radar_interfaces::msg::Header & msg)
  : msg_(msg)
  {}
  Init_Header_num_tlvs num_detected_obj(::mmwave_radar_interfaces::msg::Header::_num_detected_obj_type arg)
  {
    msg_.num_detected_obj = std::move(arg);
    return Init_Header_num_tlvs(msg_);
  }

private:
  ::mmwave_radar_interfaces::msg::Header msg_;
};

class Init_Header_time_cpu_cycles
{
public:
  explicit Init_Header_time_cpu_cycles(::mmwave_radar_interfaces::msg::Header & msg)
  : msg_(msg)
  {}
  Init_Header_num_detected_obj time_cpu_cycles(::mmwave_radar_interfaces::msg::Header::_time_cpu_cycles_type arg)
  {
    msg_.time_cpu_cycles = std::move(arg);
    return Init_Header_num_detected_obj(msg_);
  }

private:
  ::mmwave_radar_interfaces::msg::Header msg_;
};

class Init_Header_frame_number
{
public:
  explicit Init_Header_frame_number(::mmwave_radar_interfaces::msg::Header & msg)
  : msg_(msg)
  {}
  Init_Header_time_cpu_cycles frame_number(::mmwave_radar_interfaces::msg::Header::_frame_number_type arg)
  {
    msg_.frame_number = std::move(arg);
    return Init_Header_time_cpu_cycles(msg_);
  }

private:
  ::mmwave_radar_interfaces::msg::Header msg_;
};

class Init_Header_platform
{
public:
  explicit Init_Header_platform(::mmwave_radar_interfaces::msg::Header & msg)
  : msg_(msg)
  {}
  Init_Header_frame_number platform(::mmwave_radar_interfaces::msg::Header::_platform_type arg)
  {
    msg_.platform = std::move(arg);
    return Init_Header_frame_number(msg_);
  }

private:
  ::mmwave_radar_interfaces::msg::Header msg_;
};

class Init_Header_total_packet_len
{
public:
  explicit Init_Header_total_packet_len(::mmwave_radar_interfaces::msg::Header & msg)
  : msg_(msg)
  {}
  Init_Header_platform total_packet_len(::mmwave_radar_interfaces::msg::Header::_total_packet_len_type arg)
  {
    msg_.total_packet_len = std::move(arg);
    return Init_Header_platform(msg_);
  }

private:
  ::mmwave_radar_interfaces::msg::Header msg_;
};

class Init_Header_version
{
public:
  explicit Init_Header_version(::mmwave_radar_interfaces::msg::Header & msg)
  : msg_(msg)
  {}
  Init_Header_total_packet_len version(::mmwave_radar_interfaces::msg::Header::_version_type arg)
  {
    msg_.version = std::move(arg);
    return Init_Header_total_packet_len(msg_);
  }

private:
  ::mmwave_radar_interfaces::msg::Header msg_;
};

class Init_Header_magic_word
{
public:
  Init_Header_magic_word()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Header_version magic_word(::mmwave_radar_interfaces::msg::Header::_magic_word_type arg)
  {
    msg_.magic_word = std::move(arg);
    return Init_Header_version(msg_);
  }

private:
  ::mmwave_radar_interfaces::msg::Header msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::mmwave_radar_interfaces::msg::Header>()
{
  return mmwave_radar_interfaces::msg::builder::Init_Header_magic_word();
}

}  // namespace mmwave_radar_interfaces

#endif  // MMWAVE_RADAR_INTERFACES__MSG__DETAIL__HEADER__BUILDER_HPP_
