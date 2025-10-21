// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from mmwave_radar_interfaces:msg/SNRNoise.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "mmwave_radar_interfaces/msg/detail/snr_noise__rosidl_typesupport_introspection_c.h"
#include "mmwave_radar_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "mmwave_radar_interfaces/msg/detail/snr_noise__functions.h"
#include "mmwave_radar_interfaces/msg/detail/snr_noise__struct.h"


// Include directives for member types
// Member `snr`
// Member `noise`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__SNRNoise_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  mmwave_radar_interfaces__msg__SNRNoise__init(message_memory);
}

void mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__SNRNoise_fini_function(void * message_memory)
{
  mmwave_radar_interfaces__msg__SNRNoise__fini(message_memory);
}

size_t mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__size_function__SNRNoise__snr(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__get_const_function__SNRNoise__snr(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__get_function__SNRNoise__snr(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__fetch_function__SNRNoise__snr(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__get_const_function__SNRNoise__snr(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__assign_function__SNRNoise__snr(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__get_function__SNRNoise__snr(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__resize_function__SNRNoise__snr(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__size_function__SNRNoise__noise(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__get_const_function__SNRNoise__noise(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__get_function__SNRNoise__noise(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__fetch_function__SNRNoise__noise(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__get_const_function__SNRNoise__noise(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__assign_function__SNRNoise__noise(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__get_function__SNRNoise__noise(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__resize_function__SNRNoise__noise(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__SNRNoise_message_member_array[2] = {
  {
    "snr",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mmwave_radar_interfaces__msg__SNRNoise, snr),  // bytes offset in struct
    NULL,  // default value
    mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__size_function__SNRNoise__snr,  // size() function pointer
    mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__get_const_function__SNRNoise__snr,  // get_const(index) function pointer
    mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__get_function__SNRNoise__snr,  // get(index) function pointer
    mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__fetch_function__SNRNoise__snr,  // fetch(index, &value) function pointer
    mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__assign_function__SNRNoise__snr,  // assign(index, value) function pointer
    mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__resize_function__SNRNoise__snr  // resize(index) function pointer
  },
  {
    "noise",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mmwave_radar_interfaces__msg__SNRNoise, noise),  // bytes offset in struct
    NULL,  // default value
    mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__size_function__SNRNoise__noise,  // size() function pointer
    mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__get_const_function__SNRNoise__noise,  // get_const(index) function pointer
    mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__get_function__SNRNoise__noise,  // get(index) function pointer
    mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__fetch_function__SNRNoise__noise,  // fetch(index, &value) function pointer
    mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__assign_function__SNRNoise__noise,  // assign(index, value) function pointer
    mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__resize_function__SNRNoise__noise  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__SNRNoise_message_members = {
  "mmwave_radar_interfaces__msg",  // message namespace
  "SNRNoise",  // message name
  2,  // number of fields
  sizeof(mmwave_radar_interfaces__msg__SNRNoise),
  mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__SNRNoise_message_member_array,  // message members
  mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__SNRNoise_init_function,  // function to initialize message memory (memory has to be allocated)
  mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__SNRNoise_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__SNRNoise_message_type_support_handle = {
  0,
  &mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__SNRNoise_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_mmwave_radar_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mmwave_radar_interfaces, msg, SNRNoise)() {
  if (!mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__SNRNoise_message_type_support_handle.typesupport_identifier) {
    mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__SNRNoise_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &mmwave_radar_interfaces__msg__SNRNoise__rosidl_typesupport_introspection_c__SNRNoise_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
