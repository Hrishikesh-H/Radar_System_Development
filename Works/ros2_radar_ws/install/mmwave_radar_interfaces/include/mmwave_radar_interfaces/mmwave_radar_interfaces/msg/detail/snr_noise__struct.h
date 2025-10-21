// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mmwave_radar_interfaces:msg/SNRNoise.idl
// generated code does not contain a copyright notice

#ifndef MMWAVE_RADAR_INTERFACES__MSG__DETAIL__SNR_NOISE__STRUCT_H_
#define MMWAVE_RADAR_INTERFACES__MSG__DETAIL__SNR_NOISE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'snr'
// Member 'noise'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/SNRNoise in the package mmwave_radar_interfaces.
/**
  * SNR and Noise (TLV7)
 */
typedef struct mmwave_radar_interfaces__msg__SNRNoise
{
  rosidl_runtime_c__float__Sequence snr;
  rosidl_runtime_c__float__Sequence noise;
} mmwave_radar_interfaces__msg__SNRNoise;

// Struct for a sequence of mmwave_radar_interfaces__msg__SNRNoise.
typedef struct mmwave_radar_interfaces__msg__SNRNoise__Sequence
{
  mmwave_radar_interfaces__msg__SNRNoise * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mmwave_radar_interfaces__msg__SNRNoise__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MMWAVE_RADAR_INTERFACES__MSG__DETAIL__SNR_NOISE__STRUCT_H_
