// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from mmwave_radar_interfaces:msg/RawPacket.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "mmwave_radar_interfaces/msg/detail/raw_packet__rosidl_typesupport_introspection_c.h"
#include "mmwave_radar_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "mmwave_radar_interfaces/msg/detail/raw_packet__functions.h"
#include "mmwave_radar_interfaces/msg/detail/raw_packet__struct.h"


// Include directives for member types
// Member `data`
#include "std_msgs/msg/byte_multi_array.h"
// Member `data`
#include "std_msgs/msg/detail/byte_multi_array__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void mmwave_radar_interfaces__msg__RawPacket__rosidl_typesupport_introspection_c__RawPacket_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  mmwave_radar_interfaces__msg__RawPacket__init(message_memory);
}

void mmwave_radar_interfaces__msg__RawPacket__rosidl_typesupport_introspection_c__RawPacket_fini_function(void * message_memory)
{
  mmwave_radar_interfaces__msg__RawPacket__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember mmwave_radar_interfaces__msg__RawPacket__rosidl_typesupport_introspection_c__RawPacket_message_member_array[1] = {
  {
    "data",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mmwave_radar_interfaces__msg__RawPacket, data),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers mmwave_radar_interfaces__msg__RawPacket__rosidl_typesupport_introspection_c__RawPacket_message_members = {
  "mmwave_radar_interfaces__msg",  // message namespace
  "RawPacket",  // message name
  1,  // number of fields
  sizeof(mmwave_radar_interfaces__msg__RawPacket),
  mmwave_radar_interfaces__msg__RawPacket__rosidl_typesupport_introspection_c__RawPacket_message_member_array,  // message members
  mmwave_radar_interfaces__msg__RawPacket__rosidl_typesupport_introspection_c__RawPacket_init_function,  // function to initialize message memory (memory has to be allocated)
  mmwave_radar_interfaces__msg__RawPacket__rosidl_typesupport_introspection_c__RawPacket_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t mmwave_radar_interfaces__msg__RawPacket__rosidl_typesupport_introspection_c__RawPacket_message_type_support_handle = {
  0,
  &mmwave_radar_interfaces__msg__RawPacket__rosidl_typesupport_introspection_c__RawPacket_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_mmwave_radar_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mmwave_radar_interfaces, msg, RawPacket)() {
  mmwave_radar_interfaces__msg__RawPacket__rosidl_typesupport_introspection_c__RawPacket_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, ByteMultiArray)();
  if (!mmwave_radar_interfaces__msg__RawPacket__rosidl_typesupport_introspection_c__RawPacket_message_type_support_handle.typesupport_identifier) {
    mmwave_radar_interfaces__msg__RawPacket__rosidl_typesupport_introspection_c__RawPacket_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &mmwave_radar_interfaces__msg__RawPacket__rosidl_typesupport_introspection_c__RawPacket_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
