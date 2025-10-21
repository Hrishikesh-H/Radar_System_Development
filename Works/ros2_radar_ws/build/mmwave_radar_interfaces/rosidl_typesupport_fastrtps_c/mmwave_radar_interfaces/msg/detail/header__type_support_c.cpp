// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from mmwave_radar_interfaces:msg/Header.idl
// generated code does not contain a copyright notice
#include "mmwave_radar_interfaces/msg/detail/header__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "mmwave_radar_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "mmwave_radar_interfaces/msg/detail/header__struct.h"
#include "mmwave_radar_interfaces/msg/detail/header__functions.h"
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


// forward declare type support functions


using _Header__ros_msg_type = mmwave_radar_interfaces__msg__Header;

static bool _Header__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _Header__ros_msg_type * ros_message = static_cast<const _Header__ros_msg_type *>(untyped_ros_message);
  // Field name: magic_word
  {
    size_t size = 8;
    auto array_ptr = ros_message->magic_word;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: version
  {
    cdr << ros_message->version;
  }

  // Field name: total_packet_len
  {
    cdr << ros_message->total_packet_len;
  }

  // Field name: platform
  {
    cdr << ros_message->platform;
  }

  // Field name: frame_number
  {
    cdr << ros_message->frame_number;
  }

  // Field name: time_cpu_cycles
  {
    cdr << ros_message->time_cpu_cycles;
  }

  // Field name: num_detected_obj
  {
    cdr << ros_message->num_detected_obj;
  }

  // Field name: num_tlvs
  {
    cdr << ros_message->num_tlvs;
  }

  // Field name: subframe_number
  {
    cdr << ros_message->subframe_number;
  }

  return true;
}

static bool _Header__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _Header__ros_msg_type * ros_message = static_cast<_Header__ros_msg_type *>(untyped_ros_message);
  // Field name: magic_word
  {
    size_t size = 8;
    auto array_ptr = ros_message->magic_word;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: version
  {
    cdr >> ros_message->version;
  }

  // Field name: total_packet_len
  {
    cdr >> ros_message->total_packet_len;
  }

  // Field name: platform
  {
    cdr >> ros_message->platform;
  }

  // Field name: frame_number
  {
    cdr >> ros_message->frame_number;
  }

  // Field name: time_cpu_cycles
  {
    cdr >> ros_message->time_cpu_cycles;
  }

  // Field name: num_detected_obj
  {
    cdr >> ros_message->num_detected_obj;
  }

  // Field name: num_tlvs
  {
    cdr >> ros_message->num_tlvs;
  }

  // Field name: subframe_number
  {
    cdr >> ros_message->subframe_number;
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_mmwave_radar_interfaces
size_t get_serialized_size_mmwave_radar_interfaces__msg__Header(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _Header__ros_msg_type * ros_message = static_cast<const _Header__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name magic_word
  {
    size_t array_size = 8;
    auto array_ptr = ros_message->magic_word;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name version
  {
    size_t item_size = sizeof(ros_message->version);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name total_packet_len
  {
    size_t item_size = sizeof(ros_message->total_packet_len);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name platform
  {
    size_t item_size = sizeof(ros_message->platform);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name frame_number
  {
    size_t item_size = sizeof(ros_message->frame_number);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name time_cpu_cycles
  {
    size_t item_size = sizeof(ros_message->time_cpu_cycles);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name num_detected_obj
  {
    size_t item_size = sizeof(ros_message->num_detected_obj);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name num_tlvs
  {
    size_t item_size = sizeof(ros_message->num_tlvs);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name subframe_number
  {
    size_t item_size = sizeof(ros_message->subframe_number);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _Header__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_mmwave_radar_interfaces__msg__Header(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_mmwave_radar_interfaces
size_t max_serialized_size_mmwave_radar_interfaces__msg__Header(
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

  // member: magic_word
  {
    size_t array_size = 8;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: version
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: total_packet_len
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: platform
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: frame_number
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: time_cpu_cycles
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: num_detected_obj
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: num_tlvs
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: subframe_number
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = mmwave_radar_interfaces__msg__Header;
    is_plain =
      (
      offsetof(DataType, subframe_number) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _Header__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_mmwave_radar_interfaces__msg__Header(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_Header = {
  "mmwave_radar_interfaces::msg",
  "Header",
  _Header__cdr_serialize,
  _Header__cdr_deserialize,
  _Header__get_serialized_size,
  _Header__max_serialized_size
};

static rosidl_message_type_support_t _Header__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_Header,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, mmwave_radar_interfaces, msg, Header)() {
  return &_Header__type_support;
}

#if defined(__cplusplus)
}
#endif
