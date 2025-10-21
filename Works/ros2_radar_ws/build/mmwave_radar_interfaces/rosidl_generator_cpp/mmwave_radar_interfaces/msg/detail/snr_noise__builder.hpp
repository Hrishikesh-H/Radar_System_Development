// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mmwave_radar_interfaces:msg/SNRNoise.idl
// generated code does not contain a copyright notice

#ifndef MMWAVE_RADAR_INTERFACES__MSG__DETAIL__SNR_NOISE__BUILDER_HPP_
#define MMWAVE_RADAR_INTERFACES__MSG__DETAIL__SNR_NOISE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mmwave_radar_interfaces/msg/detail/snr_noise__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mmwave_radar_interfaces
{

namespace msg
{

namespace builder
{

class Init_SNRNoise_noise
{
public:
  explicit Init_SNRNoise_noise(::mmwave_radar_interfaces::msg::SNRNoise & msg)
  : msg_(msg)
  {}
  ::mmwave_radar_interfaces::msg::SNRNoise noise(::mmwave_radar_interfaces::msg::SNRNoise::_noise_type arg)
  {
    msg_.noise = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mmwave_radar_interfaces::msg::SNRNoise msg_;
};

class Init_SNRNoise_snr
{
public:
  Init_SNRNoise_snr()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SNRNoise_noise snr(::mmwave_radar_interfaces::msg::SNRNoise::_snr_type arg)
  {
    msg_.snr = std::move(arg);
    return Init_SNRNoise_noise(msg_);
  }

private:
  ::mmwave_radar_interfaces::msg::SNRNoise msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::mmwave_radar_interfaces::msg::SNRNoise>()
{
  return mmwave_radar_interfaces::msg::builder::Init_SNRNoise_snr();
}

}  // namespace mmwave_radar_interfaces

#endif  // MMWAVE_RADAR_INTERFACES__MSG__DETAIL__SNR_NOISE__BUILDER_HPP_
