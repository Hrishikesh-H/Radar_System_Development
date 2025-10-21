// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from mmwave_radar_interfaces:msg/Header.idl
// generated code does not contain a copyright notice

#ifndef MMWAVE_RADAR_INTERFACES__MSG__DETAIL__HEADER__STRUCT_HPP_
#define MMWAVE_RADAR_INTERFACES__MSG__DETAIL__HEADER__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__mmwave_radar_interfaces__msg__Header __attribute__((deprecated))
#else
# define DEPRECATED__mmwave_radar_interfaces__msg__Header __declspec(deprecated)
#endif

namespace mmwave_radar_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Header_
{
  using Type = Header_<ContainerAllocator>;

  explicit Header_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<uint8_t, 8>::iterator, uint8_t>(this->magic_word.begin(), this->magic_word.end(), 0);
      this->version = 0ul;
      this->total_packet_len = 0ul;
      this->platform = 0ul;
      this->frame_number = 0ul;
      this->time_cpu_cycles = 0ul;
      this->num_detected_obj = 0ul;
      this->num_tlvs = 0ul;
      this->subframe_number = 0ul;
    }
  }

  explicit Header_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : magic_word(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<uint8_t, 8>::iterator, uint8_t>(this->magic_word.begin(), this->magic_word.end(), 0);
      this->version = 0ul;
      this->total_packet_len = 0ul;
      this->platform = 0ul;
      this->frame_number = 0ul;
      this->time_cpu_cycles = 0ul;
      this->num_detected_obj = 0ul;
      this->num_tlvs = 0ul;
      this->subframe_number = 0ul;
    }
  }

  // field types and members
  using _magic_word_type =
    std::array<uint8_t, 8>;
  _magic_word_type magic_word;
  using _version_type =
    uint32_t;
  _version_type version;
  using _total_packet_len_type =
    uint32_t;
  _total_packet_len_type total_packet_len;
  using _platform_type =
    uint32_t;
  _platform_type platform;
  using _frame_number_type =
    uint32_t;
  _frame_number_type frame_number;
  using _time_cpu_cycles_type =
    uint32_t;
  _time_cpu_cycles_type time_cpu_cycles;
  using _num_detected_obj_type =
    uint32_t;
  _num_detected_obj_type num_detected_obj;
  using _num_tlvs_type =
    uint32_t;
  _num_tlvs_type num_tlvs;
  using _subframe_number_type =
    uint32_t;
  _subframe_number_type subframe_number;

  // setters for named parameter idiom
  Type & set__magic_word(
    const std::array<uint8_t, 8> & _arg)
  {
    this->magic_word = _arg;
    return *this;
  }
  Type & set__version(
    const uint32_t & _arg)
  {
    this->version = _arg;
    return *this;
  }
  Type & set__total_packet_len(
    const uint32_t & _arg)
  {
    this->total_packet_len = _arg;
    return *this;
  }
  Type & set__platform(
    const uint32_t & _arg)
  {
    this->platform = _arg;
    return *this;
  }
  Type & set__frame_number(
    const uint32_t & _arg)
  {
    this->frame_number = _arg;
    return *this;
  }
  Type & set__time_cpu_cycles(
    const uint32_t & _arg)
  {
    this->time_cpu_cycles = _arg;
    return *this;
  }
  Type & set__num_detected_obj(
    const uint32_t & _arg)
  {
    this->num_detected_obj = _arg;
    return *this;
  }
  Type & set__num_tlvs(
    const uint32_t & _arg)
  {
    this->num_tlvs = _arg;
    return *this;
  }
  Type & set__subframe_number(
    const uint32_t & _arg)
  {
    this->subframe_number = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    mmwave_radar_interfaces::msg::Header_<ContainerAllocator> *;
  using ConstRawPtr =
    const mmwave_radar_interfaces::msg::Header_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mmwave_radar_interfaces::msg::Header_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mmwave_radar_interfaces::msg::Header_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mmwave_radar_interfaces::msg::Header_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mmwave_radar_interfaces::msg::Header_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mmwave_radar_interfaces::msg::Header_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mmwave_radar_interfaces::msg::Header_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mmwave_radar_interfaces::msg::Header_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mmwave_radar_interfaces::msg::Header_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mmwave_radar_interfaces__msg__Header
    std::shared_ptr<mmwave_radar_interfaces::msg::Header_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mmwave_radar_interfaces__msg__Header
    std::shared_ptr<mmwave_radar_interfaces::msg::Header_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Header_ & other) const
  {
    if (this->magic_word != other.magic_word) {
      return false;
    }
    if (this->version != other.version) {
      return false;
    }
    if (this->total_packet_len != other.total_packet_len) {
      return false;
    }
    if (this->platform != other.platform) {
      return false;
    }
    if (this->frame_number != other.frame_number) {
      return false;
    }
    if (this->time_cpu_cycles != other.time_cpu_cycles) {
      return false;
    }
    if (this->num_detected_obj != other.num_detected_obj) {
      return false;
    }
    if (this->num_tlvs != other.num_tlvs) {
      return false;
    }
    if (this->subframe_number != other.subframe_number) {
      return false;
    }
    return true;
  }
  bool operator!=(const Header_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Header_

// alias to use template instance with default allocator
using Header =
  mmwave_radar_interfaces::msg::Header_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace mmwave_radar_interfaces

#endif  // MMWAVE_RADAR_INTERFACES__MSG__DETAIL__HEADER__STRUCT_HPP_
