// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mmwave_radar_interfaces:msg/Heartbeat.idl
// generated code does not contain a copyright notice

#ifndef MMWAVE_RADAR_INTERFACES__MSG__DETAIL__HEARTBEAT__STRUCT_H_
#define MMWAVE_RADAR_INTERFACES__MSG__DETAIL__HEARTBEAT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'builtin_time'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in msg/Heartbeat in the package mmwave_radar_interfaces.
/**
  * Heartbeat with timestamp and frame count
 */
typedef struct mmwave_radar_interfaces__msg__Heartbeat
{
  builtin_interfaces__msg__Time builtin_time;
  uint32_t frame_count;
} mmwave_radar_interfaces__msg__Heartbeat;

// Struct for a sequence of mmwave_radar_interfaces__msg__Heartbeat.
typedef struct mmwave_radar_interfaces__msg__Heartbeat__Sequence
{
  mmwave_radar_interfaces__msg__Heartbeat * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mmwave_radar_interfaces__msg__Heartbeat__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MMWAVE_RADAR_INTERFACES__MSG__DETAIL__HEARTBEAT__STRUCT_H_
