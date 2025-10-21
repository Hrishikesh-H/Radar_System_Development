// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from mmwave_radar_interfaces:msg/Header.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "mmwave_radar_interfaces/msg/detail/header__rosidl_typesupport_introspection_c.h"
#include "mmwave_radar_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "mmwave_radar_interfaces/msg/detail/header__functions.h"
#include "mmwave_radar_interfaces/msg/detail/header__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__Header_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  mmwave_radar_interfaces__msg__Header__init(message_memory);
}

void mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__Header_fini_function(void * message_memory)
{
  mmwave_radar_interfaces__msg__Header__fini(message_memory);
}

size_t mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__size_function__Header__magic_word(
  const void * untyped_member)
{
  (void)untyped_member;
  return 8;
}

const void * mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__get_const_function__Header__magic_word(
  const void * untyped_member, size_t index)
{
  const uint8_t * member =
    (const uint8_t *)(untyped_member);
  return &member[index];
}

void * mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__get_function__Header__magic_word(
  void * untyped_member, size_t index)
{
  uint8_t * member =
    (uint8_t *)(untyped_member);
  return &member[index];
}

void mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__fetch_function__Header__magic_word(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const uint8_t * item =
    ((const uint8_t *)
    mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__get_const_function__Header__magic_word(untyped_member, index));
  uint8_t * value =
    (uint8_t *)(untyped_value);
  *value = *item;
}

void mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__assign_function__Header__magic_word(
  void * untyped_member, size_t index, const void * untyped_value)
{
  uint8_t * item =
    ((uint8_t *)
    mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__get_function__Header__magic_word(untyped_member, index));
  const uint8_t * value =
    (const uint8_t *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__Header_message_member_array[9] = {
  {
    "magic_word",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    8,  // array size
    false,  // is upper bound
    offsetof(mmwave_radar_interfaces__msg__Header, magic_word),  // bytes offset in struct
    NULL,  // default value
    mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__size_function__Header__magic_word,  // size() function pointer
    mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__get_const_function__Header__magic_word,  // get_const(index) function pointer
    mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__get_function__Header__magic_word,  // get(index) function pointer
    mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__fetch_function__Header__magic_word,  // fetch(index, &value) function pointer
    mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__assign_function__Header__magic_word,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "version",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mmwave_radar_interfaces__msg__Header, version),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "total_packet_len",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mmwave_radar_interfaces__msg__Header, total_packet_len),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "platform",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mmwave_radar_interfaces__msg__Header, platform),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "frame_number",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mmwave_radar_interfaces__msg__Header, frame_number),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "time_cpu_cycles",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mmwave_radar_interfaces__msg__Header, time_cpu_cycles),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "num_detected_obj",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mmwave_radar_interfaces__msg__Header, num_detected_obj),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "num_tlvs",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mmwave_radar_interfaces__msg__Header, num_tlvs),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "subframe_number",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mmwave_radar_interfaces__msg__Header, subframe_number),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__Header_message_members = {
  "mmwave_radar_interfaces__msg",  // message namespace
  "Header",  // message name
  9,  // number of fields
  sizeof(mmwave_radar_interfaces__msg__Header),
  mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__Header_message_member_array,  // message members
  mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__Header_init_function,  // function to initialize message memory (memory has to be allocated)
  mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__Header_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__Header_message_type_support_handle = {
  0,
  &mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__Header_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_mmwave_radar_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mmwave_radar_interfaces, msg, Header)() {
  if (!mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__Header_message_type_support_handle.typesupport_identifier) {
    mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__Header_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &mmwave_radar_interfaces__msg__Header__rosidl_typesupport_introspection_c__Header_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
