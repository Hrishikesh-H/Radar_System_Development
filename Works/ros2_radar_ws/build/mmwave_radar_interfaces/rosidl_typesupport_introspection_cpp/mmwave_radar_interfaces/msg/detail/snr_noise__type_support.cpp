// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from mmwave_radar_interfaces:msg/SNRNoise.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "mmwave_radar_interfaces/msg/detail/snr_noise__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace mmwave_radar_interfaces
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void SNRNoise_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) mmwave_radar_interfaces::msg::SNRNoise(_init);
}

void SNRNoise_fini_function(void * message_memory)
{
  auto typed_message = static_cast<mmwave_radar_interfaces::msg::SNRNoise *>(message_memory);
  typed_message->~SNRNoise();
}

size_t size_function__SNRNoise__snr(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__SNRNoise__snr(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__SNRNoise__snr(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__SNRNoise__snr(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__SNRNoise__snr(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__SNRNoise__snr(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__SNRNoise__snr(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__SNRNoise__snr(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

size_t size_function__SNRNoise__noise(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__SNRNoise__noise(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__SNRNoise__noise(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__SNRNoise__noise(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__SNRNoise__noise(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__SNRNoise__noise(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__SNRNoise__noise(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__SNRNoise__noise(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember SNRNoise_message_member_array[2] = {
  {
    "snr",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mmwave_radar_interfaces::msg::SNRNoise, snr),  // bytes offset in struct
    nullptr,  // default value
    size_function__SNRNoise__snr,  // size() function pointer
    get_const_function__SNRNoise__snr,  // get_const(index) function pointer
    get_function__SNRNoise__snr,  // get(index) function pointer
    fetch_function__SNRNoise__snr,  // fetch(index, &value) function pointer
    assign_function__SNRNoise__snr,  // assign(index, value) function pointer
    resize_function__SNRNoise__snr  // resize(index) function pointer
  },
  {
    "noise",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mmwave_radar_interfaces::msg::SNRNoise, noise),  // bytes offset in struct
    nullptr,  // default value
    size_function__SNRNoise__noise,  // size() function pointer
    get_const_function__SNRNoise__noise,  // get_const(index) function pointer
    get_function__SNRNoise__noise,  // get(index) function pointer
    fetch_function__SNRNoise__noise,  // fetch(index, &value) function pointer
    assign_function__SNRNoise__noise,  // assign(index, value) function pointer
    resize_function__SNRNoise__noise  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers SNRNoise_message_members = {
  "mmwave_radar_interfaces::msg",  // message namespace
  "SNRNoise",  // message name
  2,  // number of fields
  sizeof(mmwave_radar_interfaces::msg::SNRNoise),
  SNRNoise_message_member_array,  // message members
  SNRNoise_init_function,  // function to initialize message memory (memory has to be allocated)
  SNRNoise_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t SNRNoise_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &SNRNoise_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace mmwave_radar_interfaces


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<mmwave_radar_interfaces::msg::SNRNoise>()
{
  return &::mmwave_radar_interfaces::msg::rosidl_typesupport_introspection_cpp::SNRNoise_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, mmwave_radar_interfaces, msg, SNRNoise)() {
  return &::mmwave_radar_interfaces::msg::rosidl_typesupport_introspection_cpp::SNRNoise_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
