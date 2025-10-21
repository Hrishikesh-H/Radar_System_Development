// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from mmwave_radar_interfaces:msg/SNRNoise.idl
// generated code does not contain a copyright notice

#ifndef MMWAVE_RADAR_INTERFACES__MSG__DETAIL__SNR_NOISE__TRAITS_HPP_
#define MMWAVE_RADAR_INTERFACES__MSG__DETAIL__SNR_NOISE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "mmwave_radar_interfaces/msg/detail/snr_noise__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace mmwave_radar_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const SNRNoise & msg,
  std::ostream & out)
{
  out << "{";
  // member: snr
  {
    if (msg.snr.size() == 0) {
      out << "snr: []";
    } else {
      out << "snr: [";
      size_t pending_items = msg.snr.size();
      for (auto item : msg.snr) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: noise
  {
    if (msg.noise.size() == 0) {
      out << "noise: []";
    } else {
      out << "noise: [";
      size_t pending_items = msg.noise.size();
      for (auto item : msg.noise) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SNRNoise & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: snr
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.snr.size() == 0) {
      out << "snr: []\n";
    } else {
      out << "snr:\n";
      for (auto item : msg.snr) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: noise
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.noise.size() == 0) {
      out << "noise: []\n";
    } else {
      out << "noise:\n";
      for (auto item : msg.noise) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SNRNoise & msg, bool use_flow_style = false)
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
  const mmwave_radar_interfaces::msg::SNRNoise & msg,
  std::ostream & out, size_t indentation = 0)
{
  mmwave_radar_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mmwave_radar_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const mmwave_radar_interfaces::msg::SNRNoise & msg)
{
  return mmwave_radar_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<mmwave_radar_interfaces::msg::SNRNoise>()
{
  return "mmwave_radar_interfaces::msg::SNRNoise";
}

template<>
inline const char * name<mmwave_radar_interfaces::msg::SNRNoise>()
{
  return "mmwave_radar_interfaces/msg/SNRNoise";
}

template<>
struct has_fixed_size<mmwave_radar_interfaces::msg::SNRNoise>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<mmwave_radar_interfaces::msg::SNRNoise>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<mmwave_radar_interfaces::msg::SNRNoise>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MMWAVE_RADAR_INTERFACES__MSG__DETAIL__SNR_NOISE__TRAITS_HPP_
