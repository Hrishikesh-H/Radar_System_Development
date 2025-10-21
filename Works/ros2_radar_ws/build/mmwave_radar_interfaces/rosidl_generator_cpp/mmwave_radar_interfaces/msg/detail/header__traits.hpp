// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from mmwave_radar_interfaces:msg/Header.idl
// generated code does not contain a copyright notice

#ifndef MMWAVE_RADAR_INTERFACES__MSG__DETAIL__HEADER__TRAITS_HPP_
#define MMWAVE_RADAR_INTERFACES__MSG__DETAIL__HEADER__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "mmwave_radar_interfaces/msg/detail/header__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace mmwave_radar_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const Header & msg,
  std::ostream & out)
{
  out << "{";
  // member: magic_word
  {
    if (msg.magic_word.size() == 0) {
      out << "magic_word: []";
    } else {
      out << "magic_word: [";
      size_t pending_items = msg.magic_word.size();
      for (auto item : msg.magic_word) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: version
  {
    out << "version: ";
    rosidl_generator_traits::value_to_yaml(msg.version, out);
    out << ", ";
  }

  // member: total_packet_len
  {
    out << "total_packet_len: ";
    rosidl_generator_traits::value_to_yaml(msg.total_packet_len, out);
    out << ", ";
  }

  // member: platform
  {
    out << "platform: ";
    rosidl_generator_traits::value_to_yaml(msg.platform, out);
    out << ", ";
  }

  // member: frame_number
  {
    out << "frame_number: ";
    rosidl_generator_traits::value_to_yaml(msg.frame_number, out);
    out << ", ";
  }

  // member: time_cpu_cycles
  {
    out << "time_cpu_cycles: ";
    rosidl_generator_traits::value_to_yaml(msg.time_cpu_cycles, out);
    out << ", ";
  }

  // member: num_detected_obj
  {
    out << "num_detected_obj: ";
    rosidl_generator_traits::value_to_yaml(msg.num_detected_obj, out);
    out << ", ";
  }

  // member: num_tlvs
  {
    out << "num_tlvs: ";
    rosidl_generator_traits::value_to_yaml(msg.num_tlvs, out);
    out << ", ";
  }

  // member: subframe_number
  {
    out << "subframe_number: ";
    rosidl_generator_traits::value_to_yaml(msg.subframe_number, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Header & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: magic_word
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.magic_word.size() == 0) {
      out << "magic_word: []\n";
    } else {
      out << "magic_word:\n";
      for (auto item : msg.magic_word) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: version
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "version: ";
    rosidl_generator_traits::value_to_yaml(msg.version, out);
    out << "\n";
  }

  // member: total_packet_len
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "total_packet_len: ";
    rosidl_generator_traits::value_to_yaml(msg.total_packet_len, out);
    out << "\n";
  }

  // member: platform
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "platform: ";
    rosidl_generator_traits::value_to_yaml(msg.platform, out);
    out << "\n";
  }

  // member: frame_number
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "frame_number: ";
    rosidl_generator_traits::value_to_yaml(msg.frame_number, out);
    out << "\n";
  }

  // member: time_cpu_cycles
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "time_cpu_cycles: ";
    rosidl_generator_traits::value_to_yaml(msg.time_cpu_cycles, out);
    out << "\n";
  }

  // member: num_detected_obj
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "num_detected_obj: ";
    rosidl_generator_traits::value_to_yaml(msg.num_detected_obj, out);
    out << "\n";
  }

  // member: num_tlvs
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "num_tlvs: ";
    rosidl_generator_traits::value_to_yaml(msg.num_tlvs, out);
    out << "\n";
  }

  // member: subframe_number
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "subframe_number: ";
    rosidl_generator_traits::value_to_yaml(msg.subframe_number, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Header & msg, bool use_flow_style = false)
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
  const mmwave_radar_interfaces::msg::Header & msg,
  std::ostream & out, size_t indentation = 0)
{
  mmwave_radar_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mmwave_radar_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const mmwave_radar_interfaces::msg::Header & msg)
{
  return mmwave_radar_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<mmwave_radar_interfaces::msg::Header>()
{
  return "mmwave_radar_interfaces::msg::Header";
}

template<>
inline const char * name<mmwave_radar_interfaces::msg::Header>()
{
  return "mmwave_radar_interfaces/msg/Header";
}

template<>
struct has_fixed_size<mmwave_radar_interfaces::msg::Header>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<mmwave_radar_interfaces::msg::Header>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<mmwave_radar_interfaces::msg::Header>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MMWAVE_RADAR_INTERFACES__MSG__DETAIL__HEADER__TRAITS_HPP_
