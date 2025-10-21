// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from mmwave_radar_interfaces:msg/Heartbeat.idl
// generated code does not contain a copyright notice

#ifndef MMWAVE_RADAR_INTERFACES__MSG__DETAIL__HEARTBEAT__TRAITS_HPP_
#define MMWAVE_RADAR_INTERFACES__MSG__DETAIL__HEARTBEAT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "mmwave_radar_interfaces/msg/detail/heartbeat__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'builtin_time'
#include "builtin_interfaces/msg/detail/time__traits.hpp"

namespace mmwave_radar_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const Heartbeat & msg,
  std::ostream & out)
{
  out << "{";
  // member: builtin_time
  {
    out << "builtin_time: ";
    to_flow_style_yaml(msg.builtin_time, out);
    out << ", ";
  }

  // member: frame_count
  {
    out << "frame_count: ";
    rosidl_generator_traits::value_to_yaml(msg.frame_count, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Heartbeat & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: builtin_time
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "builtin_time:\n";
    to_block_style_yaml(msg.builtin_time, out, indentation + 2);
  }

  // member: frame_count
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "frame_count: ";
    rosidl_generator_traits::value_to_yaml(msg.frame_count, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Heartbeat & msg, bool use_flow_style = false)
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
  const mmwave_radar_interfaces::msg::Heartbeat & msg,
  std::ostream & out, size_t indentation = 0)
{
  mmwave_radar_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mmwave_radar_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const mmwave_radar_interfaces::msg::Heartbeat & msg)
{
  return mmwave_radar_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<mmwave_radar_interfaces::msg::Heartbeat>()
{
  return "mmwave_radar_interfaces::msg::Heartbeat";
}

template<>
inline const char * name<mmwave_radar_interfaces::msg::Heartbeat>()
{
  return "mmwave_radar_interfaces/msg/Heartbeat";
}

template<>
struct has_fixed_size<mmwave_radar_interfaces::msg::Heartbeat>
  : std::integral_constant<bool, has_fixed_size<builtin_interfaces::msg::Time>::value> {};

template<>
struct has_bounded_size<mmwave_radar_interfaces::msg::Heartbeat>
  : std::integral_constant<bool, has_bounded_size<builtin_interfaces::msg::Time>::value> {};

template<>
struct is_message<mmwave_radar_interfaces::msg::Heartbeat>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MMWAVE_RADAR_INTERFACES__MSG__DETAIL__HEARTBEAT__TRAITS_HPP_
