// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from mmwave_radar_interfaces:msg/SNRNoise.idl
// generated code does not contain a copyright notice

#ifndef MMWAVE_RADAR_INTERFACES__MSG__DETAIL__SNR_NOISE__STRUCT_HPP_
#define MMWAVE_RADAR_INTERFACES__MSG__DETAIL__SNR_NOISE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__mmwave_radar_interfaces__msg__SNRNoise __attribute__((deprecated))
#else
# define DEPRECATED__mmwave_radar_interfaces__msg__SNRNoise __declspec(deprecated)
#endif

namespace mmwave_radar_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SNRNoise_
{
  using Type = SNRNoise_<ContainerAllocator>;

  explicit SNRNoise_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit SNRNoise_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _snr_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _snr_type snr;
  using _noise_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _noise_type noise;

  // setters for named parameter idiom
  Type & set__snr(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->snr = _arg;
    return *this;
  }
  Type & set__noise(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->noise = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    mmwave_radar_interfaces::msg::SNRNoise_<ContainerAllocator> *;
  using ConstRawPtr =
    const mmwave_radar_interfaces::msg::SNRNoise_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mmwave_radar_interfaces::msg::SNRNoise_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mmwave_radar_interfaces::msg::SNRNoise_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mmwave_radar_interfaces::msg::SNRNoise_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mmwave_radar_interfaces::msg::SNRNoise_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mmwave_radar_interfaces::msg::SNRNoise_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mmwave_radar_interfaces::msg::SNRNoise_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mmwave_radar_interfaces::msg::SNRNoise_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mmwave_radar_interfaces::msg::SNRNoise_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mmwave_radar_interfaces__msg__SNRNoise
    std::shared_ptr<mmwave_radar_interfaces::msg::SNRNoise_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mmwave_radar_interfaces__msg__SNRNoise
    std::shared_ptr<mmwave_radar_interfaces::msg::SNRNoise_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SNRNoise_ & other) const
  {
    if (this->snr != other.snr) {
      return false;
    }
    if (this->noise != other.noise) {
      return false;
    }
    return true;
  }
  bool operator!=(const SNRNoise_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SNRNoise_

// alias to use template instance with default allocator
using SNRNoise =
  mmwave_radar_interfaces::msg::SNRNoise_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace mmwave_radar_interfaces

#endif  // MMWAVE_RADAR_INTERFACES__MSG__DETAIL__SNR_NOISE__STRUCT_HPP_
