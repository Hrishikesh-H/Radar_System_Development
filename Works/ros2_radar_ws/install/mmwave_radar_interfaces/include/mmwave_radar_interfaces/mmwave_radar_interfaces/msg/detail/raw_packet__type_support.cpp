// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from mmwave_radar_interfaces:msg/RawPacket.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "mmwave_radar_interfaces/msg/detail/raw_packet__struct.hpp"
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

void RawPacket_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) mmwave_radar_interfaces::msg::RawPacket(_init);
}

void RawPacket_fini_function(void * message_memory)
{
  auto typed_message = static_cast<mmwave_radar_interfaces::msg::RawPacket *>(message_memory);
  typed_message->~RawPacket();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember RawPacket_message_member_array[1] = {
  {
    "data",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::ByteMultiArray>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mmwave_radar_interfaces::msg::RawPacket, data),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers RawPacket_message_members = {
  "mmwave_radar_interfaces::msg",  // message namespace
  "RawPacket",  // message name
  1,  // number of fields
  sizeof(mmwave_radar_interfaces::msg::RawPacket),
  RawPacket_message_member_array,  // message members
  RawPacket_init_function,  // function to initialize message memory (memory has to be allocated)
  RawPacket_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t RawPacket_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &RawPacket_message_members,
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
get_message_type_support_handle<mmwave_radar_interfaces::msg::RawPacket>()
{
  return &::mmwave_radar_interfaces::msg::rosidl_typesupport_introspection_cpp::RawPacket_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, mmwave_radar_interfaces, msg, RawPacket)() {
  return &::mmwave_radar_interfaces::msg::rosidl_typesupport_introspection_cpp::RawPacket_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
