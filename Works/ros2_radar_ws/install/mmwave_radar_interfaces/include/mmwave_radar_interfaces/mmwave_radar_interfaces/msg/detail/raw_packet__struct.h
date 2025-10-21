// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mmwave_radar_interfaces:msg/RawPacket.idl
// generated code does not contain a copyright notice

#ifndef MMWAVE_RADAR_INTERFACES__MSG__DETAIL__RAW_PACKET__STRUCT_H_
#define MMWAVE_RADAR_INTERFACES__MSG__DETAIL__RAW_PACKET__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'data'
#include "std_msgs/msg/detail/byte_multi_array__struct.h"

/// Struct defined in msg/RawPacket in the package mmwave_radar_interfaces.
/**
  * Raw packet bytes using standard multi-array
 */
typedef struct mmwave_radar_interfaces__msg__RawPacket
{
  std_msgs__msg__ByteMultiArray data;
} mmwave_radar_interfaces__msg__RawPacket;

// Struct for a sequence of mmwave_radar_interfaces__msg__RawPacket.
typedef struct mmwave_radar_interfaces__msg__RawPacket__Sequence
{
  mmwave_radar_interfaces__msg__RawPacket * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mmwave_radar_interfaces__msg__RawPacket__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MMWAVE_RADAR_INTERFACES__MSG__DETAIL__RAW_PACKET__STRUCT_H_
