// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from mmwave_radar_interfaces:msg/DetectedObjects.idl
// generated code does not contain a copyright notice

#ifndef MMWAVE_RADAR_INTERFACES__MSG__DETAIL__DETECTED_OBJECTS__STRUCT_HPP_
#define MMWAVE_RADAR_INTERFACES__MSG__DETAIL__DETECTED_OBJECTS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__mmwave_radar_interfaces__msg__DetectedObjects __attribute__((deprecated))
#else
# define DEPRECATED__mmwave_radar_interfaces__msg__DetectedObjects __declspec(deprecated)
#endif

namespace mmwave_radar_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct DetectedObjects_
{
  using Type = DetectedObjects_<ContainerAllocator>;

  explicit DetectedObjects_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->num_obj = 0ul;
    }
  }

  explicit DetectedObjects_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->num_obj = 0ul;
    }
  }

  // field types and members
  using _num_obj_type =
    uint32_t;
  _num_obj_type num_obj;
  using _x_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _x_type x;
  using _y_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _y_type y;
  using _z_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _z_type z;
  using _velocity_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _velocity_type velocity;

  // setters for named parameter idiom
  Type & set__num_obj(
    const uint32_t & _arg)
  {
    this->num_obj = _arg;
    return *this;
  }
  Type & set__x(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->x = _arg;
    return *this;
  }
  Type & set__y(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->y = _arg;
    return *this;
  }
  Type & set__z(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->z = _arg;
    return *this;
  }
  Type & set__velocity(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->velocity = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    mmwave_radar_interfaces::msg::DetectedObjects_<ContainerAllocator> *;
  using ConstRawPtr =
    const mmwave_radar_interfaces::msg::DetectedObjects_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<mmwave_radar_interfaces::msg::DetectedObjects_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<mmwave_radar_interfaces::msg::DetectedObjects_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      mmwave_radar_interfaces::msg::DetectedObjects_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<mmwave_radar_interfaces::msg::DetectedObjects_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      mmwave_radar_interfaces::msg::DetectedObjects_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<mmwave_radar_interfaces::msg::DetectedObjects_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<mmwave_radar_interfaces::msg::DetectedObjects_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<mmwave_radar_interfaces::msg::DetectedObjects_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__mmwave_radar_interfaces__msg__DetectedObjects
    std::shared_ptr<mmwave_radar_interfaces::msg::DetectedObjects_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__mmwave_radar_interfaces__msg__DetectedObjects
    std::shared_ptr<mmwave_radar_interfaces::msg::DetectedObjects_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const DetectedObjects_ & other) const
  {
    if (this->num_obj != other.num_obj) {
      return false;
    }
    if (this->x != other.x) {
      return false;
    }
    if (this->y != other.y) {
      return false;
    }
    if (this->z != other.z) {
      return false;
    }
    if (this->velocity != other.velocity) {
      return false;
    }
    return true;
  }
  bool operator!=(const DetectedObjects_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct DetectedObjects_

// alias to use template instance with default allocator
using DetectedObjects =
  mmwave_radar_interfaces::msg::DetectedObjects_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace mmwave_radar_interfaces

#endif  // MMWAVE_RADAR_INTERFACES__MSG__DETAIL__DETECTED_OBJECTS__STRUCT_HPP_
