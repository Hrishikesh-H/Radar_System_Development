// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from mmwave_radar_interfaces:msg/DetectedObjects.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "mmwave_radar_interfaces/msg/detail/detected_objects__rosidl_typesupport_introspection_c.h"
#include "mmwave_radar_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "mmwave_radar_interfaces/msg/detail/detected_objects__functions.h"
#include "mmwave_radar_interfaces/msg/detail/detected_objects__struct.h"


// Include directives for member types
// Member `x`
// Member `y`
// Member `z`
// Member `velocity`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__DetectedObjects_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  mmwave_radar_interfaces__msg__DetectedObjects__init(message_memory);
}

void mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__DetectedObjects_fini_function(void * message_memory)
{
  mmwave_radar_interfaces__msg__DetectedObjects__fini(message_memory);
}

size_t mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__size_function__DetectedObjects__x(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_const_function__DetectedObjects__x(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_function__DetectedObjects__x(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__fetch_function__DetectedObjects__x(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_const_function__DetectedObjects__x(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__assign_function__DetectedObjects__x(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_function__DetectedObjects__x(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__resize_function__DetectedObjects__x(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__size_function__DetectedObjects__y(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_const_function__DetectedObjects__y(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_function__DetectedObjects__y(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__fetch_function__DetectedObjects__y(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_const_function__DetectedObjects__y(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__assign_function__DetectedObjects__y(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_function__DetectedObjects__y(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__resize_function__DetectedObjects__y(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__size_function__DetectedObjects__z(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_const_function__DetectedObjects__z(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_function__DetectedObjects__z(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__fetch_function__DetectedObjects__z(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_const_function__DetectedObjects__z(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__assign_function__DetectedObjects__z(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_function__DetectedObjects__z(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__resize_function__DetectedObjects__z(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__size_function__DetectedObjects__velocity(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_const_function__DetectedObjects__velocity(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_function__DetectedObjects__velocity(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__fetch_function__DetectedObjects__velocity(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_const_function__DetectedObjects__velocity(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__assign_function__DetectedObjects__velocity(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_function__DetectedObjects__velocity(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__resize_function__DetectedObjects__velocity(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__DetectedObjects_message_member_array[5] = {
  {
    "num_obj",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mmwave_radar_interfaces__msg__DetectedObjects, num_obj),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "x",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mmwave_radar_interfaces__msg__DetectedObjects, x),  // bytes offset in struct
    NULL,  // default value
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__size_function__DetectedObjects__x,  // size() function pointer
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_const_function__DetectedObjects__x,  // get_const(index) function pointer
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_function__DetectedObjects__x,  // get(index) function pointer
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__fetch_function__DetectedObjects__x,  // fetch(index, &value) function pointer
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__assign_function__DetectedObjects__x,  // assign(index, value) function pointer
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__resize_function__DetectedObjects__x  // resize(index) function pointer
  },
  {
    "y",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mmwave_radar_interfaces__msg__DetectedObjects, y),  // bytes offset in struct
    NULL,  // default value
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__size_function__DetectedObjects__y,  // size() function pointer
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_const_function__DetectedObjects__y,  // get_const(index) function pointer
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_function__DetectedObjects__y,  // get(index) function pointer
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__fetch_function__DetectedObjects__y,  // fetch(index, &value) function pointer
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__assign_function__DetectedObjects__y,  // assign(index, value) function pointer
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__resize_function__DetectedObjects__y  // resize(index) function pointer
  },
  {
    "z",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mmwave_radar_interfaces__msg__DetectedObjects, z),  // bytes offset in struct
    NULL,  // default value
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__size_function__DetectedObjects__z,  // size() function pointer
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_const_function__DetectedObjects__z,  // get_const(index) function pointer
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_function__DetectedObjects__z,  // get(index) function pointer
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__fetch_function__DetectedObjects__z,  // fetch(index, &value) function pointer
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__assign_function__DetectedObjects__z,  // assign(index, value) function pointer
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__resize_function__DetectedObjects__z  // resize(index) function pointer
  },
  {
    "velocity",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mmwave_radar_interfaces__msg__DetectedObjects, velocity),  // bytes offset in struct
    NULL,  // default value
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__size_function__DetectedObjects__velocity,  // size() function pointer
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_const_function__DetectedObjects__velocity,  // get_const(index) function pointer
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__get_function__DetectedObjects__velocity,  // get(index) function pointer
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__fetch_function__DetectedObjects__velocity,  // fetch(index, &value) function pointer
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__assign_function__DetectedObjects__velocity,  // assign(index, value) function pointer
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__resize_function__DetectedObjects__velocity  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__DetectedObjects_message_members = {
  "mmwave_radar_interfaces__msg",  // message namespace
  "DetectedObjects",  // message name
  5,  // number of fields
  sizeof(mmwave_radar_interfaces__msg__DetectedObjects),
  mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__DetectedObjects_message_member_array,  // message members
  mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__DetectedObjects_init_function,  // function to initialize message memory (memory has to be allocated)
  mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__DetectedObjects_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__DetectedObjects_message_type_support_handle = {
  0,
  &mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__DetectedObjects_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_mmwave_radar_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mmwave_radar_interfaces, msg, DetectedObjects)() {
  if (!mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__DetectedObjects_message_type_support_handle.typesupport_identifier) {
    mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__DetectedObjects_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &mmwave_radar_interfaces__msg__DetectedObjects__rosidl_typesupport_introspection_c__DetectedObjects_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
