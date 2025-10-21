// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from mmwave_radar_interfaces:msg/RawPacket.idl
// generated code does not contain a copyright notice

#ifndef MMWAVE_RADAR_INTERFACES__MSG__DETAIL__RAW_PACKET__TRAITS_HPP_
#define MMWAVE_RADAR_INTERFACES__MSG__DETAIL__RAW_PACKET__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "mmwave_radar_interfaces/msg/detail/raw_packet__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'data'
#include "std_msgs/msg/detail/byte_multi_array__traits.hpp"

namespace mmwave_radar_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const RawPacket & msg,
  std::ostream & out)
{
  out << "{";
  // member: data
  {
    out << "data: ";
    to_flow_style_yaml(msg.data, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const RawPacket & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: data
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "data:\n";
    to_block_style_yaml(msg.data, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const RawPacket & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace mmwave_radar_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use mmwave_radar_interfaces::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const mmwave_radar_interfaces::msg::RawPacket & msg,
  std::ostream & out, size_t indentation = 0)
{
  mmwave_radar_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mmwave_radar_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const mmwave_radar_interfaces::msg::RawPacket & msg)
{
  return mmwave_radar_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<mmwave_radar_interfaces::msg::RawPacket>()
{
  return "mmwave_radar_interfaces::msg::RawPacket";
}

template<>
inline const char * name<mmwave_radar_interfaces::msg::RawPacket>()
{
  return "mmwave_radar_interfaces/msg/RawPacket";
}

template<>
struct has_fixed_size<mmwave_radar_interfaces::msg::RawPacket>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::ByteMultiArray>::value> {};

template<>
struct has_bounded_size<mmwave_radar_interfaces::msg::RawPacket>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::ByteMultiArray>::value> {};

template<>
struct is_message<mmwave_radar_interfaces::msg::RawPacket>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MMWAVE_RADAR_INTERFACES__MSG__DETAIL__RAW_PACKET__TRAITS_HPP_
