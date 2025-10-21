// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from mmwave_radar_interfaces:msg/SNRNoise.idl
// generated code does not contain a copyright notice

#ifndef MMWAVE_RADAR_INTERFACES__MSG__DETAIL__SNR_NOISE__FUNCTIONS_H_
#define MMWAVE_RADAR_INTERFACES__MSG__DETAIL__SNR_NOISE__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "mmwave_radar_interfaces/msg/rosidl_generator_c__visibility_control.h"

#include "mmwave_radar_interfaces/msg/detail/snr_noise__struct.h"

/// Initialize msg/SNRNoise message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * mmwave_radar_interfaces__msg__SNRNoise
 * )) before or use
 * mmwave_radar_interfaces__msg__SNRNoise__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_mmwave_radar_interfaces
bool
mmwave_radar_interfaces__msg__SNRNoise__init(mmwave_radar_interfaces__msg__SNRNoise * msg);

/// Finalize msg/SNRNoise message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mmwave_radar_interfaces
void
mmwave_radar_interfaces__msg__SNRNoise__fini(mmwave_radar_interfaces__msg__SNRNoise * msg);

/// Create msg/SNRNoise message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * mmwave_radar_interfaces__msg__SNRNoise__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mmwave_radar_interfaces
mmwave_radar_interfaces__msg__SNRNoise *
mmwave_radar_interfaces__msg__SNRNoise__create();

/// Destroy msg/SNRNoise message.
/**
 * It calls
 * mmwave_radar_interfaces__msg__SNRNoise__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mmwave_radar_interfaces
void
mmwave_radar_interfaces__msg__SNRNoise__destroy(mmwave_radar_interfaces__msg__SNRNoise * msg);

/// Check for msg/SNRNoise message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mmwave_radar_interfaces
bool
mmwave_radar_interfaces__msg__SNRNoise__are_equal(const mmwave_radar_interfaces__msg__SNRNoise * lhs, const mmwave_radar_interfaces__msg__SNRNoise * rhs);

/// Copy a msg/SNRNoise message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_mmwave_radar_interfaces
bool
mmwave_radar_interfaces__msg__SNRNoise__copy(
  const mmwave_radar_interfaces__msg__SNRNoise * input,
  mmwave_radar_interfaces__msg__SNRNoise * output);

/// Initialize array of msg/SNRNoise messages.
/**
 * It allocates the memory for the number of elements and calls
 * mmwave_radar_interfaces__msg__SNRNoise__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_mmwave_radar_interfaces
bool
mmwave_radar_interfaces__msg__SNRNoise__Sequence__init(mmwave_radar_interfaces__msg__SNRNoise__Sequence * array, size_t size);

/// Finalize array of msg/SNRNoise messages.
/**
 * It calls
 * mmwave_radar_interfaces__msg__SNRNoise__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mmwave_radar_interfaces
void
mmwave_radar_interfaces__msg__SNRNoise__Sequence__fini(mmwave_radar_interfaces__msg__SNRNoise__Sequence * array);

/// Create array of msg/SNRNoise messages.
/**
 * It allocates the memory for the array and calls
 * mmwave_radar_interfaces__msg__SNRNoise__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_mmwave_radar_interfaces
mmwave_radar_interfaces__msg__SNRNoise__Sequence *
mmwave_radar_interfaces__msg__SNRNoise__Sequence__create(size_t size);

/// Destroy array of msg/SNRNoise messages.
/**
 * It calls
 * mmwave_radar_interfaces__msg__SNRNoise__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_mmwave_radar_interfaces
void
mmwave_radar_interfaces__msg__SNRNoise__Sequence__destroy(mmwave_radar_interfaces__msg__SNRNoise__Sequence * array);

/// Check for msg/SNRNoise message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_mmwave_radar_interfaces
bool
mmwave_radar_interfaces__msg__SNRNoise__Sequence__are_equal(const mmwave_radar_interfaces__msg__SNRNoise__Sequence * lhs, const mmwave_radar_interfaces__msg__SNRNoise__Sequence * rhs);

/// Copy an array of msg/SNRNoise messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_mmwave_radar_interfaces
bool
mmwave_radar_interfaces__msg__SNRNoise__Sequence__copy(
  const mmwave_radar_interfaces__msg__SNRNoise__Sequence * input,
  mmwave_radar_interfaces__msg__SNRNoise__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // MMWAVE_RADAR_INTERFACES__MSG__DETAIL__SNR_NOISE__FUNCTIONS_H_
