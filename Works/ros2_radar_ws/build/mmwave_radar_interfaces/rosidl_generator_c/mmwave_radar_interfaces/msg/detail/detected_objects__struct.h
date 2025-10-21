// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mmwave_radar_interfaces:msg/DetectedObjects.idl
// generated code does not contain a copyright notice

#ifndef MMWAVE_RADAR_INTERFACES__MSG__DETAIL__DETECTED_OBJECTS__STRUCT_H_
#define MMWAVE_RADAR_INTERFACES__MSG__DETAIL__DETECTED_OBJECTS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'x'
// Member 'y'
// Member 'z'
// Member 'velocity'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/DetectedObjects in the package mmwave_radar_interfaces.
/**
  * Detected Objects (TLV1)
 */
typedef struct mmwave_radar_interfaces__msg__DetectedObjects
{
  uint32_t num_obj;
  rosidl_runtime_c__float__Sequence x;
  rosidl_runtime_c__float__Sequence y;
  rosidl_runtime_c__float__Sequence z;
  rosidl_runtime_c__float__Sequence velocity;
} mmwave_radar_interfaces__msg__DetectedObjects;

// Struct for a sequence of mmwave_radar_interfaces__msg__DetectedObjects.
typedef struct mmwave_radar_interfaces__msg__DetectedObjects__Sequence
{
  mmwave_radar_interfaces__msg__DetectedObjects * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mmwave_radar_interfaces__msg__DetectedObjects__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MMWAVE_RADAR_INTERFACES__MSG__DETAIL__DETECTED_OBJECTS__STRUCT_H_
