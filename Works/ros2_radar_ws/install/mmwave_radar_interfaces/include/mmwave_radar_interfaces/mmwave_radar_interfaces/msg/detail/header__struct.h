// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mmwave_radar_interfaces:msg/Header.idl
// generated code does not contain a copyright notice

#ifndef MMWAVE_RADAR_INTERFACES__MSG__DETAIL__HEADER__STRUCT_H_
#define MMWAVE_RADAR_INTERFACES__MSG__DETAIL__HEADER__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/Header in the package mmwave_radar_interfaces.
/**
  * Radar Packet Header
 */
typedef struct mmwave_radar_interfaces__msg__Header
{
  uint8_t magic_word[8];
  uint32_t version;
  uint32_t total_packet_len;
  uint32_t platform;
  uint32_t frame_number;
  uint32_t time_cpu_cycles;
  uint32_t num_detected_obj;
  uint32_t num_tlvs;
  uint32_t subframe_number;
} mmwave_radar_interfaces__msg__Header;

// Struct for a sequence of mmwave_radar_interfaces__msg__Header.
typedef struct mmwave_radar_interfaces__msg__Header__Sequence
{
  mmwave_radar_interfaces__msg__Header * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mmwave_radar_interfaces__msg__Header__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MMWAVE_RADAR_INTERFACES__MSG__DETAIL__HEADER__STRUCT_H_
