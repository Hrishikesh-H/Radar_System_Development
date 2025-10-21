// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from mmwave_radar_interfaces:msg/SNRNoise.idl
// generated code does not contain a copyright notice
#include "mmwave_radar_interfaces/msg/detail/snr_noise__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "mmwave_radar_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "mmwave_radar_interfaces/msg/detail/snr_noise__struct.h"
#include "mmwave_radar_interfaces/msg/detail/snr_noise__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif

#include "rosidl_runtime_c/primitives_sequence.h"  // noise, snr
#include "rosidl_runtime_c/primitives_sequence_functions.h"  // noise, snr

// forward declare type support functions


using _SNRNoise__ros_msg_type = mmwave_radar_interfaces__msg__SNRNoise;

static bool _SNRNoise__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _SNRNoise__ros_msg_type * ros_message = static_cast<const _SNRNoise__ros_msg_type *>(untyped_ros_message);
  // Field name: snr
  {
    size_t size = ros_message->snr.size;
    auto array_ptr = ros_message->snr.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: noise
  {
    size_t size = ros_message->noise.size;
    auto array_ptr = ros_message->noise.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  return true;
}

static bool _SNRNoise__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _SNRNoise__ros_msg_type * ros_message = static_cast<_SNRNoise__ros_msg_type *>(untyped_ros_message);
  // Field name: snr
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->snr.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->snr);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->snr, size)) {
      fprintf(stderr, "failed to create array for field 'snr'");
      return false;
    }
    auto array_ptr = ros_message->snr.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: noise
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->noise.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->noise);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->noise, size)) {
      fprintf(stderr, "failed to create array for field 'noise'");
      return false;
    }
    auto array_ptr = ros_message->noise.data;
    cdr.deserializeArray(array_ptr, size);
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_mmwave_radar_interfaces
size_t get_serialized_size_mmwave_radar_interfaces__msg__SNRNoise(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _SNRNoise__ros_msg_type * ros_message = static_cast<const _SNRNoise__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name snr
  {
    size_t array_size = ros_message->snr.size;
    auto array_ptr = ros_message->snr.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name noise
  {
    size_t array_size = ros_message->noise.size;
    auto array_ptr = ros_message->noise.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _SNRNoise__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_mmwave_radar_interfaces__msg__SNRNoise(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_mmwave_radar_interfaces
size_t max_serialized_size_mmwave_radar_interfaces__msg__SNRNoise(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // member: snr
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: noise
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = mmwave_radar_interfaces__msg__SNRNoise;
    is_plain =
      (
      offsetof(DataType, noise) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _SNRNoise__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_mmwave_radar_interfaces__msg__SNRNoise(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_SNRNoise = {
  "mmwave_radar_interfaces::msg",
  "SNRNoise",
  _SNRNoise__cdr_serialize,
  _SNRNoise__cdr_deserialize,
  _SNRNoise__get_serialized_size,
  _SNRNoise__max_serialized_size
};

static rosidl_message_type_support_t _SNRNoise__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_SNRNoise,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, mmwave_radar_interfaces, msg, SNRNoise)() {
  return &_SNRNoise__type_support;
}

#if defined(__cplusplus)
}
#endif
